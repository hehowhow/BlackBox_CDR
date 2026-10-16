"""Trusted-side, training-list-only coarse source-item release for LIST.

The target learner must receive only ``release_payload(...)``.  The ID-to-code
mapping, original query lists and all outputs labelled ``trusted`` stay on the
trusted side.  Codes reduce direct item resolution; this is neither differential
privacy nor a claim to conceal interests, identities or whole-list fingerprints.

This is a single collaborative code, not a residual semantic-ID hierarchy.
Only the explicitly supplied training users fit its vocabulary and geometry.
Validation/test users are transformed with that frozen mapping afterwards.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


COARSE_SID_DEFAULTS = {
    "code_count": 256,
    "svd_dim": 32,
    "svd_iterations": 5,
    "min_bucket_items": 8,
    "kmeans_batch_size": 4096,
    "kmeans_n_init": 3,
    "kmeans_max_iter": 100,
    "fit_threads": 2,
}


def array_hash(array: np.ndarray) -> str:
    """Stable shape/dtype/content digest for an audit artifact."""
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.shape).encode("ascii"))
    h.update(value.dtype.str.encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def coarse_sid_config(cfg: dict | None = None) -> dict:
    options = dict(COARSE_SID_DEFAULTS)
    if cfg is not None:
        unknown = set(cfg) - set(options)
        if unknown:
            raise ValueError(f"Unknown coarse-SID settings: {sorted(unknown)}")
        options.update(cfg)
    for key, value in options.items():
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        options[key] = int(value)
    return options


def _integer_array(values: np.ndarray, name: str, ndim: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != ndim or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a {ndim}-D integer array")
    result = values.astype(np.int64, copy=False)
    if (result < 0).any():
        raise ValueError(f"{name} must contain nonnegative integers")
    return result


def _validated_users(users: np.ndarray, count: int, name: str) -> np.ndarray:
    users = _integer_array(users, name, 1)
    if not len(users) or users.max() >= count:
        raise ValueError(f"{name} must be nonempty valid row indices")
    if len(np.unique(users)) != len(users):
        raise ValueError(f"{name} must not contain duplicates")
    return np.sort(users)


def _bucket_statistics(counts: np.ndarray) -> dict:
    return {
        "count": int(len(counts)),
        "minimum": int(counts.min()),
        "median": float(np.median(counts)),
        "mean": float(counts.mean()),
        "maximum": int(counts.max()),
    }


def _merge_small_buckets(
    labels: np.ndarray, features: np.ndarray, minimum: int,
) -> np.ndarray:
    """Merge small training-item clusters to closest sufficiently large ones.

    Distances use training-item centroid geometry, never held-out lists.  In
    degenerate cases with no qualifying cluster, accumulate the largest cluster
    and nearest neighbours until one qualifies, then apply the same rule.
    """
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64)
    count = np.bincount(labels)
    centers = np.vstack([features[labels == i].mean(0) for i in range(len(count))])
    eligible = np.flatnonzero(count >= minimum)
    if not len(eligible):
        anchor = int(np.argmax(count))
        distance = np.square(centers - centers[anchor]).sum(1)
        for other in np.argsort(distance, kind="stable"):
            if count[anchor] >= minimum:
                break
            if other == anchor:
                continue
            total = count[anchor] + count[other]
            centers[anchor] = (
                centers[anchor] * count[anchor] + centers[other] * count[other]
            ) / total
            labels[labels == other] = anchor
            count[anchor], count[other] = total, 0
        eligible = np.array([anchor], dtype=np.int64)
    small = np.flatnonzero((count > 0) & (count < minimum))
    # Assignment is fixed to eligible training centroids; no label depends on
    # the order in which small clusters happen to be processed.
    for cluster in small:
        distance = np.square(centers[eligible] - centers[cluster]).sum(1)
        labels[labels == cluster] = eligible[int(np.argmin(distance))]
    _, compact = np.unique(labels, return_inverse=True)
    assert np.bincount(compact).min() >= minimum
    return compact.astype(np.int64) + 1  # zero is target-side padding only


def fit_release(
    raw_ids: np.ndarray,
    fit_users: np.ndarray,
    seed: int,
    cfg: dict | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Fit on training Top-K lists and release frozen coarse codes for all rows.

    ``raw_ids[u, r]`` is a nonnegative opaque source ID at rank ``r``.  Raw ID
    zero is a real item, not padding.  ``fit_users`` contains the *shared* users
    in this phase's target training fold; target ratings are not accepted.

    Return ``(released_codes, trusted_artifacts, audit)``.  Codes are contiguous
    integers 1..audit['vocab_size']; callers may add zero rows for users without
    source queries.  Every fitted code has at least ``min_bucket_items`` unique
    training items.  Items absent from the fitting lists always use the largest
    training-item bucket.  This rule is fixed before viewing held-out rows, so
    neither held-out co-occurrence nor its vocabulary affects any fitted code.

    Unknown-to-largest-bucket mapping intentionally avoids creating a uniquely
    identifying "unknown" bucket when only one unseen ID is queried.  It also
    means unseen IDs cannot be distinguished from that bucket's known items.
    No SID embedding, reverse map, feature or original ID is a target payload.
    """
    from scipy import sparse
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    from threadpoolctl import threadpool_limits

    options = coarse_sid_config(cfg)
    raw_ids = _integer_array(raw_ids, "raw_ids", 2)
    if min(raw_ids.shape) < 1:
        raise ValueError("raw_ids must have at least one user and rank")
    users = _validated_users(fit_users, len(raw_ids), "fit_users")
    fit_ids = np.ascontiguousarray(raw_ids[users])
    vocabulary, row_index, frequency = np.unique(
        fit_ids, return_inverse=True, return_counts=True,
    )
    minimum = options["min_bucket_items"]
    if len(vocabulary) < minimum:
        raise ValueError("Too few training items to satisfy minimum bucket size")
    k = fit_ids.shape[1]
    rank_weights = (1.0 / np.log2(np.arange(k, dtype=np.float64) + 2)).astype(np.float32)
    rank_weights /= rank_weights.sum()
    columns = np.repeat(np.arange(len(users), dtype=np.int64), k)
    matrix = sparse.coo_matrix(
        (np.tile(rank_weights, len(users)), (row_index.reshape(-1), columns)),
        shape=(len(vocabulary), len(users)), dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    normalize(matrix, norm="l2", axis=1, copy=False)
    dimension = min(options["svd_dim"], min(matrix.shape))
    fit_seed = int(seed)
    with threadpool_limits(limits=options["fit_threads"]):
        if min(matrix.shape) == 1:
            u, singular, _ = np.linalg.svd(matrix.toarray(), full_matrices=False)
            features = (u[:, :dimension] * singular[:dimension]).astype(np.float32)
            explained_variance = None
        else:
            svd = TruncatedSVD(
                n_components=dimension,
                n_iter=options["svd_iterations"],
                random_state=fit_seed,
            )
            features = svd.fit_transform(matrix).astype(np.float32)
            explained_variance = float(np.nan_to_num(svd.explained_variance_ratio_).sum())
        normalize(features, norm="l2", axis=1, copy=False)
        distinct_features = len(np.unique(features, axis=0))
        initial_clusters = min(
            options["code_count"], max(1, len(vocabulary) // minimum), distinct_features,
        )
        if initial_clusters == 1:
            labels = np.zeros(len(vocabulary), dtype=np.int64)
        else:
            quantizer = MiniBatchKMeans(
                n_clusters=initial_clusters,
                random_state=fit_seed + 1,
                n_init=options["kmeans_n_init"],
                batch_size=options["kmeans_batch_size"],
                max_iter=options["kmeans_max_iter"],
                reassignment_ratio=0.01,
            )
            labels = quantizer.fit_predict(features).astype(np.int64)
        fit_codes = _merge_small_buckets(labels, features, minimum)

    counts = np.bincount(fit_codes)[1:]
    code_count = len(counts)
    unknown_code = int(np.argmax(counts)) + 1
    # Everything above this line depends solely on fit_ids, fit_users, seed and
    # cfg.  Only now do we inspect the other rows to apply the frozen release.
    all_ids, full_inverse = np.unique(raw_ids, return_inverse=True)
    positions = np.searchsorted(vocabulary, all_ids)
    known = positions < len(vocabulary)
    known[known] &= vocabulary[positions[known]] == all_ids[known]
    all_codes = np.full(len(all_ids), unknown_code, dtype=np.int64)
    all_codes[known] = fit_codes[positions[known]]
    released = all_codes[full_inverse].reshape(raw_ids.shape)
    release_counts = np.bincount(all_codes, minlength=code_count + 1)[1:]
    unknown_positions = ~known[full_inverse]
    trusted = {
        "source_ids": all_ids,
        "source_codes": all_codes,
        "fit_source_ids": vocabulary,
        "fit_source_codes": fit_codes,
        "fit_source_frequency": frequency.astype(np.int64),
        "fit_user_ids": users,
        "rank_weights": rank_weights,
    }
    audit = {
        "status": "PASS",
        "algorithm": "training-list rank-weighted item-user SVD + single shared code",
        "settings": options,
        "seed": fit_seed,
        "fit_user_count": int(len(users)),
        "fit_item_count": int(len(vocabulary)),
        "fit_source_hash": array_hash(fit_ids),
        "fit_user_hash": array_hash(users),
        "fit_mapping_hash": array_hash(np.column_stack([vocabulary, fit_codes])),
        "released_codes_hash": array_hash(released),
        "svd_effective_dim": int(dimension),
        "svd_explained_variance_ratio_sum": explained_variance,
        "initial_cluster_count": int(initial_clusters),
        "code_count": int(code_count),
        "vocab_size": int(code_count),
        "padding_code": 0,
        "unknown_code": unknown_code,
        "unknown_rule": "always largest TRAINING unique-item bucket; no held-out-dependent decision",
        "fit_bucket_items": _bucket_statistics(counts),
        "release_bucket_items": _bucket_statistics(release_counts),
        "release_user_count": int(len(raw_ids)),
        "release_k": int(k),
        "release_unique_source_items": int(len(all_ids)),
        "release_unknown_unique_items": int((~known).sum()),
        "release_unknown_positions": int(unknown_positions.sum()),
        "release_unknown_position_fraction": float(unknown_positions.mean()),
        "target_ratings_used_for_codes": False,
        "held_out_users_used_for_fitting": False,
        "rank_order_preserved": True,
        "duplicate_codes_preserved": True,
        "codes_are_coarse_to_fine_prefixes": False,
        "privacy_claim": "lower direct item resolution only; no DP or user anonymity guarantee",
    }
    return released, trusted, audit


def release_payload(released_codes: np.ndarray) -> dict[str, np.ndarray]:
    """Allowlisted target artifact: aligned users, codes, and public dimensions.

    The legacy key ``item_ids`` now contains code IDs, never raw source IDs.
    This helper deliberately accepts no trusted mapping or original-ID input.
    """
    codes = _integer_array(released_codes, "released_codes", 2)
    if not codes.size or codes.min() < 1:
        raise ValueError("Released queries must contain positive nonpadding codes")
    unique = np.unique(codes)
    if not np.array_equal(unique, np.arange(1, int(codes.max()) + 1)):
        raise ValueError("Code vocabulary must be contiguous starting from one")
    return {
        "user_ids": np.arange(len(codes), dtype=np.int64),
        "item_ids": codes.copy(),
        "vocab_size": np.asarray(int(codes.max()), dtype=np.int64),
        "code_count": np.asarray(int(codes.max()), dtype=np.int64),
    }


def recovery_audit(
    raw_ids: np.ndarray,
    released_codes: np.ndarray,
    trusted: dict[str, np.ndarray],
    eval_users: np.ndarray,
) -> dict[str, Any]:
    """Trusted evaluation of a deliberately limited item-recovery adversary.

    The attacker is given the entire code-to-item map and training-list item
    frequencies.  For each received code it guesses the most frequent training
    item in that bucket (ties: smallest raw ID); it does NOT get eval original
    IDs, user histories or item metadata.  Eval originals are used only to score
    guesses.  This is a simple attack, not an upper bound against other attacks.

    The uniform-candidate number is explicitly a hypothetical uniform-guess
    reference, NOT the actual identification probability or a privacy guarantee.
    No target ratings are used, including in this evaluation.
    """
    raw = _integer_array(raw_ids, "raw_ids", 2)
    codes = _integer_array(released_codes, "released_codes", 2)
    if raw.shape != codes.shape:
        raise ValueError("Raw and released query shapes differ")
    users = _validated_users(eval_users, len(raw), "eval_users")
    source_ids = trusted["fit_source_ids"]
    fit_codes = trusted["fit_source_codes"]
    frequency = trusted["fit_source_frequency"]
    code_count = int(codes.max())
    guesses = np.full(code_count + 1, -1, dtype=np.int64)
    for code in range(1, code_count + 1):
        members = np.flatnonzero(fit_codes == code)
        if not len(members):
            raise ValueError("Released code has no training candidate")
        chosen = members[int(np.argmax(frequency[members]))]
        guesses[code] = source_ids[chosen]
    expected = codes[users]
    correct = guesses[expected] == raw[users]
    candidates = np.bincount(trusted["source_codes"], minlength=code_count + 1)
    training_candidates = np.bincount(fit_codes, minlength=code_count + 1)
    weights = 1.0 / np.log2(np.arange(raw.shape[1], dtype=np.float64) + 2)
    weights /= weights.sum()
    fit_vocabulary = np.sort(source_ids)
    pos = np.searchsorted(fit_vocabulary, raw[users])
    seen = pos < len(fit_vocabulary)
    seen[seen] &= fit_vocabulary[pos[seen]] == raw[users][seen]
    return {
        "status": "PASS",
        "attack": "per-code training item-frequency MAP; ties smallest source ID",
        "attacker_knows_full_item_mapping": True,
        "attacker_knows_training_item_frequencies": True,
        "attacker_has_eval_original_lists": False,
        "not_a_bound_on_other_attacks": True,
        "eval_user_count": int(len(users)),
        "eval_user_hash": array_hash(users),
        "raw_id_direct_recovery_rate": 1.0,
        "frequency_map_position_recovery_rate": float(correct.mean()),
        "frequency_map_rank_weighted_recovery_rate": float((correct @ weights).mean()),
        "frequency_map_whole_list_recovery_rate": float(correct.all(1).mean()),
        "frequency_map_seen_item_recovery_rate": float(correct[seen].mean()) if seen.any() else None,
        "eval_seen_item_position_fraction": float(seen.mean()),
        "minimum_training_candidates_per_observed_code": int(training_candidates[expected].min()),
        "mean_candidate_items_per_position": float(candidates[expected].mean()),
        "hypothetical_uniform_candidate_guess_rate": float((1.0 / candidates[expected]).mean()),
        "privacy_guarantee": False,
    }
