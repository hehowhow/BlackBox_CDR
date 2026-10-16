"""Trusted-side, frequency-only reconstruction checks for coarse LIST releases.

This module never reads files, target ratings, model checkpoints, or GPUs.  Its
input raw IDs are source *recommendation outputs*, not listening histories.
Only ``fit_users`` supply attack frequencies.  ``eval_users`` are held-out row
indices used for scoring, and must be disjoint from the fitting rows.

An optional ``trusted_artifacts`` mapping may contain ``catalog_item_ids`` and
``catalog_codes`` arrays.  Without these, the largest reported catalogue is
explicitly only the items observed anywhere in ``raw_ids``.  No raw IDs or
reconstruction candidate lists are returned in the aggregate JSON report.
"""

from collections.abc import Mapping

import numpy as np


def _users(value, size, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer row index array")
    value = value.astype(np.int64, copy=False)
    if len(value) == 0 or np.any(value < 0) or np.any(value >= size):
        raise ValueError(f"{name} must contain nonempty in-range row indices")
    if len(np.unique(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicate rows")
    return value


def _bucket_summary(item_codes):
    _, counts = np.unique(item_codes, return_counts=True)
    return {
        "items": int(len(item_codes)),
        "codes": int(len(counts)),
        "minimum_items_per_code": int(counts.min()),
        "median_items_per_code": float(np.median(counts)),
        "p95_items_per_code": float(np.quantile(counts, .95)),
        "maximum_items_per_code": int(counts.max()),
        "singleton_codes": int(np.sum(counts == 1)),
        "items_in_singleton_codes_fraction": float(np.sum(counts == 1) / len(item_codes)),
    }


def _sequence_summary(codes):
    _, counts = np.unique(codes, axis=0, return_counts=True)
    return {
        "rows": int(len(codes)),
        "distinct_sequences": int(len(counts)),
        "distinct_sequence_fraction": float(len(counts) / len(codes)),
        "rows_with_unique_sequence_fraction": float(np.sum(counts == 1) / len(codes)),
        "meaning": "Exact ordered code-list uniqueness within these rows; not identity re-identification.",
    }


def _top_candidates(code_indices, item_indices, frequencies, global_frequencies, code_count):
    """Return up to five positive-frequency candidates per code, deterministically.

    Item indices follow sorted raw IDs, so the final tie-break does not depend
    on how often an evaluation item occurs.  For rank-conditioned estimates,
    global training frequency breaks equal rank-frequency ties.
    """
    order = np.lexsort((item_indices, -global_frequencies, -frequencies, code_indices))
    groups = code_indices[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(groups)) + 1, len(groups)]
    top = np.zeros((code_count, 5), dtype=np.int64)
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        selected = order[begin:min(begin + 5, end)]
        top[groups[begin], :len(selected)] = item_indices[selected] + 1
    return top


def evaluate_release(raw_ids, released_codes, fit_users, eval_users, trusted_artifacts=None):
    """Score fixed frequency attacks on held-out source-list item identities.

    ``raw_ids`` and ``released_codes`` are same-shaped, integer ``[users, K]``
    arrays without padding.  A raw item must always map to the same released
    code.  The attacker is restricted to fitting-row empirical frequencies;
    known mappings/side information or stronger attacks can do better.

    Code-MAP ranks candidates by their fitting-row slot frequency.  Rank-MAP
    uses fitting-row frequency at that exact rank, breaking ties by global
    fitting frequency, then sorted item ID.  It falls back to code-MAP only
    when no fitting observation exists for the entire (code, rank) pair.
    Top-5 contains up to five positive-frequency candidates, without invented
    unseen-item guesses or test-informed completion of candidate lists.
    """
    raw_ids = np.asarray(raw_ids)
    released_codes = np.asarray(released_codes)
    if (raw_ids.ndim != 2 or raw_ids.shape != released_codes.shape
            or min(raw_ids.shape) == 0):
        raise ValueError("raw_ids and released_codes must be nonempty, matching [users, K] arrays")
    if raw_ids.dtype.kind not in "iu" or released_codes.dtype.kind not in "iu":
        raise ValueError("raw_ids and released_codes must contain integer identifiers")
    fit_users = _users(fit_users, len(raw_ids), "fit_users")
    eval_users = _users(eval_users, len(raw_ids), "eval_users")
    if np.intersect1d(fit_users, eval_users).size:
        raise ValueError("fit_users and eval_users must be disjoint")

    # All-row catalogue information below is descriptive/lookup-only.  No
    # all-row frequency is used for any attack candidate or tie-break.
    all_items, first, inverse = np.unique(raw_ids.ravel(), return_index=True, return_inverse=True)
    item_codes = released_codes.ravel()[first]
    inverse = inverse.reshape(raw_ids.shape)
    if not np.array_equal(item_codes[inverse], released_codes):
        raise ValueError("A raw item has inconsistent released codes")
    code_values = np.unique(item_codes)
    item_code_indices = np.searchsorted(code_values, item_codes)
    fit_item_indices, fit_counts = np.unique(inverse[fit_users], return_counts=True)
    fit_global_counts = np.zeros(len(all_items), dtype=np.int64)
    fit_global_counts[fit_item_indices] = fit_counts
    fit_code_indices = item_code_indices[fit_item_indices]
    top_global = _top_candidates(fit_code_indices, fit_item_indices, fit_counts,
                                 fit_counts, len(code_values))

    truth = inverse[eval_users] + 1
    eval_code_indices = np.searchsorted(code_values, released_codes[eval_users])
    global_hits1 = global_hits5 = rank_hits1 = rank_hits5 = 0
    rank_fallback_slots = unseen_code_slots = 0
    for rank in range(raw_ids.shape[1]):
        rank_items, rank_counts = np.unique(inverse[fit_users, rank], return_counts=True)
        rank_top = _top_candidates(item_code_indices[rank_items], rank_items, rank_counts,
                                   fit_global_counts[rank_items], len(code_values))
        absent = rank_top[:, 0] == 0
        rank_top[absent] = top_global[absent]
        code_index = eval_code_indices[:, rank]
        global_predictions = top_global[code_index]
        rank_predictions = rank_top[code_index]
        expected = truth[:, rank, None]
        global_hits1 += int(np.sum(global_predictions[:, :1] == expected))
        global_hits5 += int(np.any(global_predictions == expected, axis=1).sum())
        rank_hits1 += int(np.sum(rank_predictions[:, :1] == expected))
        rank_hits5 += int(np.any(rank_predictions == expected, axis=1).sum())
        rank_fallback_slots += int(absent[code_index].sum())
        unseen_code_slots += int((global_predictions[:, 0] == 0).sum())

    slots = int(truth.size)
    unseen_slots = fit_global_counts[truth - 1] == 0
    eval_unique = np.unique(truth - 1)
    report = {
        "status": "PASS",
        "scope": "Source recommendation-list item identity; not source interaction history or user identity.",
        "attack_training": "Only fit_users empirical frequencies; no evaluation frequencies, target ratings, or model selection.",
        "fit_users": int(len(fit_users)),
        "eval_users": int(len(eval_users)),
        "list_length": int(raw_ids.shape[1]),
        "evaluation_slots": slots,
        "raw_id_release": {"exact_item_top1": 1., "exact_item_top5": 1.},
        "code_frequency_attack": {
            "exact_item_top1": float(global_hits1 / slots),
            "exact_item_top5": float(global_hits5 / slots),
        },
        "code_rank_frequency_attack": {
            "exact_item_top1": float(rank_hits1 / slots),
            "exact_item_top5": float(rank_hits5 / slots),
            "fallback_to_code_frequency_fraction": float(rank_fallback_slots / slots),
        },
        "evaluation_unseen_code_slot_fraction": float(unseen_code_slots / slots),
        "evaluation_items_absent_from_fit": {
            "slot_fraction": float(unseen_slots.mean()),
            "unique_item_fraction": float(np.mean(fit_global_counts[eval_unique] == 0)),
        },
        "fit_item_buckets": _bucket_summary(item_codes[fit_item_indices]),
        "all_queried_item_buckets": _bucket_summary(item_codes),
        "all_queried_catalogue_scope": "Unique raw IDs appearing anywhere in supplied source query lists; not necessarily the full source catalogue.",
        "evaluation_code_sequences": _sequence_summary(released_codes[eval_users]),
        "all_queried_code_sequences": _sequence_summary(released_codes),
        "limitations": [
            "These are two restricted empirical attacks, not an upper bound on reconstruction capability.",
            "Shared codes and bucket sizes do not establish user k-anonymity, differential privacy, or uniform candidate probabilities.",
            "Point and Gaussian models receiving the same codes share this release-level result.",
        ],
    }
    if trusted_artifacts is not None:
        if not isinstance(trusted_artifacts, Mapping):
            raise ValueError("trusted_artifacts must be a mapping when provided")
        has_items = "catalog_item_ids" in trusted_artifacts
        has_codes = "catalog_codes" in trusted_artifacts
        if has_items != has_codes:
            raise ValueError("Provide both catalog_item_ids and catalog_codes, or neither")
        if has_items:
            catalog_ids = np.asarray(trusted_artifacts["catalog_item_ids"])
            catalog_codes = np.asarray(trusted_artifacts["catalog_codes"])
            if (catalog_ids.ndim != 1 or catalog_ids.shape != catalog_codes.shape
                    or not len(catalog_ids) or catalog_ids.dtype.kind not in "iu"
                    or catalog_codes.dtype.kind not in "iu"):
                raise ValueError("Catalogue IDs and codes must be nonempty matching integer vectors")
            ids, starts, back = np.unique(catalog_ids, return_index=True, return_inverse=True)
            codes = catalog_codes[starts]
            if not np.array_equal(codes[back], catalog_codes):
                raise ValueError("Catalogue mapping is inconsistent")
            locations = np.searchsorted(ids, all_items)
            if np.any(locations >= len(ids)) or not np.array_equal(ids[locations], all_items):
                raise ValueError("Provided catalogue must include every queried item")
            if not np.array_equal(codes[locations], item_codes):
                raise ValueError("Catalogue mapping disagrees with released codes")
            report["provided_catalogue_item_buckets"] = _bucket_summary(codes)
            report["provided_catalogue_scope"] = "Trusted caller-supplied catalogue; used only for descriptive bucket statistics."
    return report


def run_selftest():
    """Small hand-computable attack, fallback, leakage, and mapping checks."""
    import json

    raw = np.array([[10, 20, 30], [10, 20, 40], [20, 10, 40],
                    [10, 40, 50], [20, 40, 10]], dtype=np.int64)
    codes = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2],
                      [1, 2, 1], [1, 2, 1]], dtype=np.int64)
    fit = np.array([0, 1, 2])
    held = np.array([3, 4])
    out = evaluate_release(raw, codes, fit, held)
    # Code1 ties globally: lower itemID10 wins. Code2 predicts40.
    assert out["code_frequency_attack"]["exact_item_top1"] == 4 / 6
    assert out["code_frequency_attack"]["exact_item_top5"] == 5 / 6
    assert out["code_rank_frequency_attack"]["exact_item_top1"] == 4 / 6
    assert out["code_rank_frequency_attack"]["fallback_to_code_frequency_fraction"] == 4 / 6
    assert out["evaluation_items_absent_from_fit"]["slot_fraction"] == 1 / 6
    assert out["fit_item_buckets"]["minimum_items_per_code"] == 2
    assert out["all_queried_item_buckets"]["maximum_items_per_code"] == 3
    assert out["evaluation_code_sequences"]["rows_with_unique_sequence_fraction"] == 0.
    assert out["evaluation_code_sequences"]["distinct_sequence_fraction"] == .5
    json.dumps(out, allow_nan=False)

    # Rank conditioning really changes the predictor when training evidence
    # differs by position, rather than merely relabeling the global attack.
    raw2 = np.array([[1, 2], [1, 2], [1, 2]])
    code2 = np.ones_like(raw2)
    rank_test = evaluate_release(raw2, code2, np.array([0, 1]), np.array([2]))
    assert rank_test["code_frequency_attack"]["exact_item_top1"] == .5
    assert rank_test["code_rank_frequency_attack"]["exact_item_top1"] == 1.

    # Repeated held-out IDs must not train the attack. With code1's fitting
    # majority unchanged, duplicating item2-only held-out users remains zero.
    altered = np.array([[1, 1], [1, 2], [2, 2], [2, 2]])
    stable = evaluate_release(altered, np.ones_like(altered), np.array([0, 1]), np.array([2, 3]))
    assert stable["code_frequency_attack"]["exact_item_top1"] == 0.
    assert stable["code_rank_frequency_attack"]["exact_item_top1"] == 0.

    # Entirely unseen code has no empirical training candidate, never an
    # evaluation-informed guess. Valid production releases merge unknown
    # items into a preselected trained bucket instead of using this case.
    unknown = evaluate_release(np.array([[1], [2]]), np.array([[1], [2]]), np.array([0]), np.array([1]))
    assert unknown["code_frequency_attack"]["exact_item_top1"] == 0.
    assert unknown["evaluation_unseen_code_slot_fraction"] == 1.

    extended = evaluate_release(raw, codes, fit, held, {
        "catalog_item_ids": np.array([10, 20, 30, 40, 50, 60]),
        "catalog_codes": np.array([1, 1, 2, 2, 1, 2]),
    })
    assert extended["provided_catalogue_item_buckets"]["items"] == 6
    assert extended["code_frequency_attack"] == out["code_frequency_attack"]

    bad_codes = codes.copy()
    bad_codes[-1, -1] = 2
    rejected = 0
    for args in [(raw, bad_codes, fit, held), (raw, codes, fit, np.array([0, 3]))]:
        try:
            evaluate_release(*args)
        except ValueError:
            rejected += 1
    assert rejected == 2
    return {"status": "PASS", "frequency_attacks": True, "rank_conditioning": True,
            "fallback": True, "fit_only_attack_statistics": True,
            "catalogue_statistics_do_not_train_attacks": True,
            "inconsistent_mapping_and_overlap_rejected": True, "json_serializable": True}


if __name__ == "__main__":
    import json
    print(json.dumps(run_selftest(), indent=2))
