"""CPU tests of trusted coarse-SID fit/release separation and recovery scoring."""
import json

import numpy as np

from list_coarse_sid_release import fit_release, recovery_audit, release_payload


def main():
    rng = np.random.RandomState(719)
    # Training IDs 0..127 include raw ID zero.  The held-out list contains a
    # single unseen item, so the fallback cannot form a singleton code bucket.
    raw = np.vstack([rng.choice(128, 32, replace=False) for _ in range(40)])
    raw[:4] = np.arange(128).reshape(4, 32)
    raw[30:, 0] = 10000
    train = np.arange(30, dtype=np.int64)
    test = np.arange(30, 40, dtype=np.int64)
    cfg = dict(code_count=8, svd_dim=8, min_bucket_items=8, fit_threads=1)
    codes, trusted, audit = fit_release(raw, train, 73, cfg)
    assert codes.shape == raw.shape and codes.dtype == np.int64
    assert audit["fit_item_count"] == 128 and audit["release_unknown_unique_items"] == 1
    assert audit["fit_bucket_items"]["minimum"] >= 8
    assert (codes[30:, 0] == audit["unknown_code"]).all()
    assert audit["vocab_size"] > 1
    assert 0 in trusted["fit_source_ids"] and (codes > 0).all()
    assert not np.isin(10000, trusted["fit_source_ids"])

    # Fit must be independent of held-out identities, co-occurrence and even
    # held-out vocabulary size.  Every held-out position becomes a new item.
    changed = raw.copy()
    changed[test] = np.arange(len(test) * raw.shape[1]).reshape(len(test), -1) + 20000
    changed_codes, changed_trusted, changed_audit = fit_release(changed, train[::-1], 73, cfg)
    np.testing.assert_array_equal(codes[train], changed_codes[train])
    for key in ["fit_source_ids", "fit_source_codes", "fit_source_frequency", "fit_user_ids"]:
        np.testing.assert_array_equal(trusted[key], changed_trusted[key])
    for key in ["fit_source_hash", "fit_user_hash", "fit_mapping_hash", "unknown_code", "vocab_size"]:
        assert audit[key] == changed_audit[key], key
    assert (changed_codes[test] == audit["unknown_code"]).all()

    # Deterministic lookup, rank preservation, and no deduplication of codes.
    mapping = dict(zip(trusted["source_ids"], trusted["source_codes"]))
    manual = np.vectorize(mapping.__getitem__, otypes=[np.int64])(raw)
    np.testing.assert_array_equal(manual, codes)
    assert any(len(np.unique(row)) < len(row) for row in codes)
    payload = release_payload(codes)
    assert set(payload) == {"user_ids", "item_ids", "vocab_size", "code_count"}
    np.testing.assert_array_equal(payload["user_ids"], np.arange(len(raw)))
    np.testing.assert_array_equal(payload["item_ids"], codes)
    assert int(payload["vocab_size"]) == audit["vocab_size"]
    assert not np.shares_memory(payload["item_ids"], codes)

    # Recovery uses train-frequency estimates. Independently recompute the
    # attacker and make sure changing eval ground truth changes only scoring.
    scored = recovery_audit(raw, codes, trusted, test)
    guessed = {}
    for code in np.unique(codes):
        candidates = np.flatnonzero(trusted["fit_source_codes"] == code)
        winner = candidates[np.argmax(trusted["fit_source_frequency"][candidates])]
        guessed[code] = trusted["fit_source_ids"][winner]
    independent = np.vectorize(guessed.__getitem__, otypes=[np.int64])(codes[test])
    expected_rate = float((independent == raw[test]).mean())
    assert scored["frequency_map_position_recovery_rate"] == expected_rate
    assert scored["minimum_training_candidates_per_observed_code"] >= 8
    assert 0 <= scored["frequency_map_position_recovery_rate"] < 1
    assert 0 < scored["hypothetical_uniform_candidate_guess_rate"] <= 1 / 8

    # Degenerate one-user fit is still well-defined and never exposes a
    # singleton even though its item-user features are indistinguishable.
    small = np.arange(16, dtype=np.int64).reshape(1, 16)
    small_codes, _, small_audit = fit_release(small, np.array([0]), 19, cfg)
    assert (small_codes == 1).all() and small_audit["fit_bucket_items"]["minimum"] == 16
    invalid_inputs = [
        (raw.astype(np.float64), train),
        (raw, np.array([], dtype=np.int64)),
        (raw, np.array([0, 0])),
        (raw, np.array([len(raw)])),
        (np.arange(7, dtype=np.int64).reshape(1, -1), np.array([0])),
    ]
    for invalid_raw, invalid_users in invalid_inputs:
        try:
            fit_release(invalid_raw, invalid_users, 73, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input silently accepted")
    print(json.dumps({
        "status": "PASS", "device": "cpu", "target_ratings_used": False,
        "held_out_mutation_does_not_change_fit": True,
        "unknown_singleton_merged_by_train_only_rule": True,
        "raw_id_zero_is_not_padding": True,
        "public_payload_allowlist": sorted(payload),
        "rank_order_and_duplicates_preserved": True,
        "deterministic_training_mapping": True,
        "candidate_bucket_minimum": audit["fit_bucket_items"]["minimum"],
        "degenerate_fit_checked": True,
        "recovery_independently_recomputed": True,
        "recovery": scored,
    }, indent=2))


if __name__ == "__main__":
    main()
