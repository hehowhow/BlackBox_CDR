"""LIST route B: target-code generation supplies training-only supervision.

The rating path is exactly ``ResidualList``.  A small decoder predicts three
collaborative target-item codes followed by the observed rating-residual sign.
The target codes are fitted separately in each training phase, using only the
target training rows explicitly supplied to ``fit_target_codes``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from small_task_comparison import ResidualList


TARGET_SID_DEFAULTS = {
    "target_sid_levels": 3,
    "target_sid_codebook_size": 64,
    "target_sid_svd_dim": 32,
    "target_sid_svd_iterations": 5,
    "target_sid_kmeans_batch_size": 4096,
    "target_sid_kmeans_n_init": 3,
    "target_sid_kmeans_max_iter": 100,
    "target_sid_fit_threads": 4,
    "target_sid_aux_hidden": 64,
    "target_sid_aux_weight": 0.1,
}


def target_sid_config(cfg: dict) -> dict:
    """Read only this route's settings; baseline LIST settings stay untouched."""
    return {k: cfg.get(k, v) for k, v in TARGET_SID_DEFAULTS.items()}


def _array_hash(array: np.ndarray, dtype: str) -> str:
    value = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    h = hashlib.sha256()
    h.update(str(value.shape).encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def fit_target_codes(
    data: np.ndarray,
    target_items: int,
    cfg: dict,
    seed: int,
    phase: str | None = None,
) -> dict[str, Any]:
    """Fit a fixed collaborative residual-quantization vocabulary on train only.

    ``data`` contains [user_id, item_id, rating] from ONE phase's training fold.
    There is deliberately no validation/test argument and no filesystem access.
    Target-only training users are permitted: their ratings are target-domain
    training information, not additional source-domain information.

    The sparse matrix has items as rows and *training* users as columns. Its
    entries are rating minus that phase's item mean; rows are L2-normalized.
    SVD32 gives collaborative item features, followed by three residual K-means
    quantizers. No temporal meaning is assigned to rows or to code positions.

    Returns numpy arrays plus JSON-safe ``metadata``. Save the arrays in NPZ
    and metadata in JSON. Unseen target items have all codes -1; they never
    contribute auxiliary labels. Their rating fallback is the runner's normal
    unseen-item mean policy.
    """
    from scipy import sparse
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    from threadpoolctl import threadpool_limits

    options = target_sid_config(cfg)
    levels = int(options["target_sid_levels"])
    code_count = int(options["target_sid_codebook_size"])
    dimension = int(options["target_sid_svd_dim"])
    if min(levels, code_count, dimension, target_items) <= 0:
        raise ValueError("Code dimensions and target item count must be positive")
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 3 or not len(data):
        raise ValueError("Training data must be a nonempty Nx3 array")
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite training data")
    pairs = data[:, :2].astype(np.int64)
    if not np.equal(data[:, :2], pairs).all() or np.any(pairs < 0):
        raise ValueError("Expected nonnegative integer user/item IDs")
    if pairs[:, 1].max() >= target_items:
        raise ValueError("Training item outside target item vocabulary")
    ratings = data[:, 2].astype(np.float64)
    users, user_columns = np.unique(pairs[:, 0], return_inverse=True)
    counts = np.bincount(pairs[:, 1], minlength=target_items)
    sums = np.bincount(pairs[:, 1], weights=ratings, minlength=target_items)
    seen = counts > 0
    means = np.full(target_items, ratings.mean(), dtype=np.float64)
    means[seen] = sums[seen] / counts[seen]
    residual = (ratings - means[pairs[:, 1]]).astype(np.float32)
    matrix = sparse.coo_matrix(
        (residual, (pairs[:, 1], user_columns)),
        shape=(target_items, len(users)), dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix = normalize(matrix, norm="l2", axis=1, copy=False)
    features = np.zeros((target_items, dimension), dtype=np.float32)
    codes = np.full((target_items, levels), -1, dtype=np.int64)
    centers = np.zeros((levels, code_count, dimension), dtype=np.float32)
    stage_cluster_counts: list[int] = []
    stage_reconstruction_mse: list[float] = []
    # Fitting never touches PyTorch RNG or the model's initialization stream.
    fit_seed = int(seed) + 47000
    effective_svd_dim = 0
    explained_variance_ratio_sum = 0.0
    with threadpool_limits(limits=int(options["target_sid_fit_threads"])):
        if matrix.nnz:
            effective_svd_dim = min(dimension, min(matrix.shape))
            if min(matrix.shape) == 1:
                # Tiny/degenerate cases cannot use sklearn's >=2-feature SVD.
                value = matrix.toarray()
                u, s, _ = np.linalg.svd(value, full_matrices=False)
                projected = u[:, :effective_svd_dim] * s[:effective_svd_dim]
            else:
                svd = TruncatedSVD(
                    n_components=effective_svd_dim,
                    n_iter=int(options["target_sid_svd_iterations"]),
                    random_state=fit_seed,
                )
                projected = svd.fit_transform(matrix)
                explained_variance_ratio_sum = float(
                    np.nan_to_num(svd.explained_variance_ratio_).sum()
                )
            features[:, :effective_svd_dim] = projected.astype(np.float32)
            # Normalize once before residual quantization, not after each stage.
            normalize(features, norm="l2", axis=1, copy=False)
        current = features[seen].copy()
        for level in range(levels):
            # Avoid duplicate-center warnings for zero/very-small toy matrices.
            distinct = len(np.unique(current, axis=0))
            effective_clusters = min(code_count, len(current), distinct)
            if effective_clusters == 1:
                level_centers = current.mean(axis=0, keepdims=True)
                assigned = np.zeros(len(current), dtype=np.int64)
            else:
                quantizer = MiniBatchKMeans(
                    n_clusters=effective_clusters,
                    random_state=fit_seed + level + 1,
                    n_init=int(options["target_sid_kmeans_n_init"]),
                    batch_size=int(options["target_sid_kmeans_batch_size"]),
                    max_iter=int(options["target_sid_kmeans_max_iter"]),
                    reassignment_ratio=0.01,
                )
                assigned = quantizer.fit_predict(current).astype(np.int64)
                level_centers = quantizer.cluster_centers_.astype(np.float32)
            codes[seen, level] = assigned
            centers[level, :effective_clusters] = level_centers
            current -= level_centers[assigned]
            stage_cluster_counts.append(effective_clusters)
            stage_reconstruction_mse.append(float(np.square(current).mean()))

    metadata = {
        "status": "PASS",
        "phase": phase,
        "seed": int(seed),
        "fit_seed": fit_seed,
        "settings": options,
        "fit_rows": int(len(data)),
        "fit_users": int(len(users)),
        "fit_items": int(seen.sum()),
        "target_items": int(target_items),
        "training_rows_sha256": _array_hash(data, "<f8"),
        "training_pairs_sha256": _array_hash(pairs, "<i8"),
        "training_users_sha256": _array_hash(users, "<i8"),
        "training_items_sha256": _array_hash(np.flatnonzero(seen), "<i8"),
        "item_means_sha256": _array_hash(means, "<f8"),
        "codes_sha256": _array_hash(codes, "<i8"),
        "features_sha256": _array_hash(features, "<f4"),
        "effective_svd_dim": int(effective_svd_dim),
        "svd_explained_variance_ratio_sum": explained_variance_ratio_sum,
        "residual_matrix_shape": list(matrix.shape),
        "residual_matrix_nnz": int(matrix.nnz),
        "stage_cluster_counts": stage_cluster_counts,
        "stage_reconstruction_mse": stage_reconstruction_mse,
        "matrix": "training item x compact training user; r - training item mean; item-row L2 normalization",
        "quantization": "SVD32 then feature-row L2 normalization then three residual MiniBatchKMeans codebooks",
        "fit_scope": "only explicitly supplied current-phase target training ratings; includes allowed target-only users",
        "unseen_target_code": -1,
        "uses_source_history_text_scores_or_model_parameters": False,
        "validation_or_test_records_supplied": False,
    }
    return {
        "codes": codes,
        "centers": centers,
        "features": features,
        "seen": seen,
        "fit_users": users,
        "fit_item_counts": counts,
        "fit_item_means": means,
        "metadata": metadata,
    }


class TargetCodeResidual(ResidualList):
    """Base LIST plus a training-only autoregressive collaborative-code decoder."""

    def __init__(self, lists, vocab, items, base, cfg, target_codes):
        super().__init__(lists, vocab, items, base, cfg)
        options = target_sid_config(cfg)
        self.sid_levels = int(options["target_sid_levels"])
        self.sid_codebook_size = int(options["target_sid_codebook_size"])
        self.aux_weight = float(options["target_sid_aux_weight"])
        if self.aux_weight < 0:
            raise ValueError("Auxiliary weight must be nonnegative")
        codes = torch.as_tensor(target_codes, dtype=torch.long, device=base.device)
        if tuple(codes.shape) != (items, self.sid_levels):
            raise ValueError("Target code table shape does not match model settings")
        valid = (codes >= 0).all(dim=1)
        invalid = (codes == -1).all(dim=1)
        if not (valid | invalid).all() or (codes >= self.sid_codebook_size).any():
            raise ValueError("Codes must be complete SID rows or all -1 for unseen items")
        self.register_buffer("target_codes", codes.clone())
        self.register_buffer("target_codes_valid", valid)
        hidden = int(options["target_sid_aux_hidden"])
        # Keep the base parameter initialization AND following global RNG stream
        # identical to a baseline constructor. Auxiliary init uses a forked RNG.
        with torch.random.fork_rng(devices=[]):
            self.aux_context = nn.Linear(cfg["dim"], hidden)
            self.aux_cell = nn.GRUCell(cfg["dim"], hidden)
            self.aux_token_embeddings = nn.ModuleList([
                nn.Embedding(self.sid_codebook_size, cfg["dim"])
                for _ in range(self.sid_levels)
            ])
            self.aux_bos = nn.Parameter(torch.empty(cfg["dim"]))
            nn.init.normal_(self.aux_bos, std=0.1)
            self.aux_heads = nn.ModuleList([
                nn.Linear(hidden, self.sid_codebook_size)
                for _ in range(self.sid_levels)
            ] + [nn.Linear(hidden, 3)])

    def list_vectors(self, users):
        rows = self.lists[users]
        h = self.source_items(rows, per_sample_weights=self.rank_weights.expand_as(rows))
        return self.encoder(h) * (rows[:, :1] != 0)

    def teacher_forced_logits(self, z, codes):
        """p(c1|z), p(c2|z,c1), p(c3|z,c1,c2), p(sign|z,c1,c2,c3)."""
        hidden = torch.tanh(self.aux_context(z))
        previous = self.aux_bos.expand(len(z), -1)
        logits = []
        for step, head in enumerate(self.aux_heads):
            hidden = self.aux_cell(previous, hidden)
            logits.append(head(hidden))
            if step < self.sid_levels:
                previous = self.aux_token_embeddings[step](codes[:, step])
        return logits

    def training_loss(self, users, items, ratings):
        # Share this one stochastic encoder pass between regression and decoder.
        z = self.list_vectors(users)
        prediction = self.base[items] + (z * self.target_items(items)).sum(-1)
        mse = (prediction - ratings).square().mean()
        eligible = (self.lists[users, 0] != 0) & self.target_codes_valid[items]
        eligible_count = eligible.sum()
        # Empty code-bearing minibatches occur for target-only users. Their base
        # MSE is retained, while no decoder target or gradient is fabricated.
        if bool(eligible.any()):
            codes = self.target_codes[items[eligible]]
            sign = torch.sign(ratings[eligible] - self.base[items[eligible]]).long() + 1
            logits = self.teacher_forced_logits(z[eligible], codes)
            losses = [F.cross_entropy(logits[j], codes[:, j]) for j in range(self.sid_levels)]
            losses.append(F.cross_entropy(logits[-1], sign))
            eligible_ce = torch.stack(losses).mean()
            # Match the regression's full-minibatch denominator. Target-only
            # rows contribute zero rather than amplifying eta for rare overlap
            # rows by an inverse overlap fraction.
            eligible_fraction = eligible_count.to(mse.dtype) / len(users)
            auxiliary = eligible_ce * eligible_fraction
            code_ce = torch.stack(losses[:-1]).mean() * eligible_fraction
            sign_ce = losses[-1] * eligible_fraction
        else:
            auxiliary = mse.new_zeros(())
            eligible_ce = mse.new_zeros(())
            code_ce = mse.new_zeros(())
            sign_ce = mse.new_zeros(())
        total = mse + self.aux_weight * auxiliary
        return total, {
            "mse": mse.detach(),
            "aux_ce": auxiliary.detach(),
            "eligible_aux_ce": eligible_ce.detach(),
            "code_ce": code_ce.detach(),
            "sign_ce": sign_ce.detach(),
            "aux_rows": eligible_count.detach(),
            "total": total.detach(),
        }


def selftest() -> dict:
    """Tiny CPU checks for train-only fitting, base parity and auxiliary gradients."""
    cfg = {"dim": 32, "hidden": 64, "dropout": 0.1}
    rows = np.array([
        [0, 0, 5], [0, 1, 1], [0, 2, 4],
        [1, 0, 3], [1, 1, 4], [1, 2, 2],
        [2, 0, 1], [2, 1, 5], [2, 2, 3],
        [3, 0, 4], [3, 1, 2], [3, 2, 5],
    ], dtype=np.float64)
    unrelated_eval = np.array([[99, 3, 1.0]])
    first = fit_target_codes(rows, 4, cfg, 2020, phase="select")
    unrelated_eval[:, 2] = 5.0
    second = fit_target_codes(rows.copy(), 4, cfg, 2020, phase="select")
    for key in ["codes", "features", "centers", "fit_users", "fit_item_means"]:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["metadata"] == second["metadata"]
    assert (first["codes"][3] == -1).all()
    assert 99 not in first["fit_users"]
    changed = rows.copy()
    changed[0, 2] = 2.0
    third = fit_target_codes(changed, 4, cfg, 2020, phase="select")
    assert first["metadata"]["training_rows_sha256"] != third["metadata"]["training_rows_sha256"]
    assert not np.array_equal(first["features"], third["features"])
    flat = fit_target_codes(np.array([[0, 0, 4.], [1, 0, 4.]]), 2, cfg, 2020)
    assert np.isfinite(flat["features"]).all() and (flat["codes"][0] == 0).all()

    lists = torch.tensor([[1, 2], [2, 1], [0, 0], [1, 1]])
    base = torch.tensor(first["fit_item_means"], dtype=torch.float32)
    torch.manual_seed(71)
    original = ResidualList(lists, 2, 4, base, cfg)
    original_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(71)
    model = TargetCodeResidual(lists, 2, 4, base, cfg, first["codes"])
    torch.testing.assert_close(torch.random.get_rng_state(), original_rng, rtol=0, atol=0)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    users = torch.tensor([0, 1, 2, 3])
    items = torch.tensor([0, 1, 2, 0])
    ratings = torch.tensor([5., 2., 3., 4.])
    model.eval()
    original.eval()
    torch.testing.assert_close(model(users, items), original(users, items), rtol=0, atol=0)
    torch.testing.assert_close(model(users, items), base[items], rtol=0, atol=0)
    # Regression uses exactly the baseline stochastic encoder pass. Auxiliary
    # layers have no dropout and therefore consume none of that RNG stream.
    with torch.no_grad():
        original.encoder[-1].weight.fill_(0.02)
        model.encoder[-1].weight.copy_(original.encoder[-1].weight)
    original.train()
    model.train()
    torch.manual_seed(812)
    expected_mse = (original(users, items) - ratings).square().mean()
    expected_train_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(812)
    _, same_pass = model.training_loss(users, items, ratings)
    torch.testing.assert_close(same_pass["mse"], expected_mse, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_train_rng, rtol=0, atol=0)
    with torch.no_grad():
        model.encoder[-1].weight.zero_()
    # Isolate auxiliary-only gradients, rather than relying on MSE gradients.
    model.eval()
    active = torch.tensor([0, 1, 3])
    active_codes = model.target_codes[items[active]]
    active_logits = model.teacher_forced_logits(model.list_vectors(users[active]), active_codes)
    aux_only = torch.stack([
        F.cross_entropy(active_logits[j], active_codes[:, j])
        for j in range(model.sid_levels)
    ]).mean()
    aux_only.backward()
    assert model.encoder[-1].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.train()
    loss, stat = model.training_loss(users, items, ratings)
    loss.backward()
    assert stat["aux_rows"].item() == 3
    torch.testing.assert_close(stat["aux_ce"], stat["eligible_aux_ce"] * 0.75, rtol=0, atol=0)
    assert model.encoder[-1].weight.grad.abs().sum() > 0
    assert model.aux_context.weight.grad is not None
    assert model.aux_cell.weight_ih.grad.abs().sum() > 0
    assert not model.target_codes.requires_grad
    model.zero_grad(set_to_none=True)
    only_empty, stat_empty = model.training_loss(users[2:3], items[2:3], ratings[2:3])
    assert stat_empty["aux_rows"].item() == 0 and stat_empty["aux_ce"].item() == 0
    only_empty.backward()
    assert model.aux_cell.weight_ih.grad is None
    with torch.no_grad():
        model.encoder[-1].bias.fill_(1.)
    assert model(users, items)[2] == base[items[2]]
    # Rating inference cannot change when the training-only decoder changes.
    model.eval()
    before = model(users, items).detach().clone()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("aux_"):
                parameter.add_(17.)
    torch.testing.assert_close(model(users, items), before, rtol=0, atol=0)
    # Each step may see preceding codes, never its current/future true code.
    fixture_z = torch.randn(2, 32)
    fixture_codes = torch.tensor([[1, 2, 3], [1, 2, 3]])
    logits1 = model.teacher_forced_logits(fixture_z, fixture_codes)
    fixture_codes[:, 2] = 9
    logits2 = model.teacher_forced_logits(fixture_z, fixture_codes)
    for step in range(3):
        torch.testing.assert_close(logits1[step], logits2[step], rtol=0, atol=0)
    return dict(
        status="PASS", train_only_fit_deterministic=True,
        training_ratings_influence_collaborative_features=True,
        no_eval_membership=True, unseen_target_codes_negative=True,
        same_base_parameters_and_rng=True, same_zero_init_rating_predictions=True,
        same_main_training_mse_and_dropout_rng=True, full_batch_aux_normalization=True,
        no_list_stays_at_mean=True, auxiliary_gradients_reach_encoder=True,
        no_list_rows_excluded_from_aux=True, no_future_token_teacher_forcing=True,
        decoder_ignored_at_rating_inference=True, zero_residual_matrix_handled=True,
        defaults=TARGET_SID_DEFAULTS,
    )


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2, sort_keys=True))
