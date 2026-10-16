"""Route A: target-supervised compositional source codes for residual LIST.

Only the source-ID lookup changes. Each item emits the sum of three selected
32-D code vectors; there is no item-specific continuous residual in that
representation. The original item-specific 32-D vectors remain as learnable
*selectors*, so this experiment does not claim training parameter compression.

The target rating MSE supplies straight-through gradients to the selectors,
selection keys and code values. A small minibatch code-balance penalty helps
avoid collapse. Neither fitting nor initialization needs source model weights,
scores, histories, text, or any held-out target labels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from small_task_comparison import ResidualList


SOURCE_CODE_DEFAULTS = {
    "books": 3,
    "codes_per_book": 64,
    "temperature": 0.5,
    "balance_weight": 0.01,
    "value_initial_std": 0.1,
}


class CompositionalSourceLookup(nn.Module):
    """EmbeddingBag-compatible hard compositional lookup.

    Assignment keys and emitted values are separate to avoid making a change
    in rating-vector magnitude directly change the nearest-key geometry.
    Cosine keys and the item selector live in the same 32-D space. They are
    learned only by downstream training and the code-use regularizer.
    """

    def __init__(self, original_weight, settings):
        super().__init__()
        self.books = int(settings["books"])
        self.codes_per_book = int(settings["codes_per_book"])
        self.temperature = float(settings["temperature"])
        if self.books < 1 or self.codes_per_book < 2 or self.temperature <= 0:
            raise ValueError("Invalid source-code configuration")
        self.dim = original_weight.shape[1]
        self.assignment_vectors = nn.Embedding.from_pretrained(
            original_weight.detach().clone(), freeze=False, padding_idx=0
        )
        self.selection_keys = nn.Parameter(
            original_weight.new_empty(self.books, self.codes_per_book, self.dim)
        )
        self.code_values = nn.Parameter(torch.empty_like(self.selection_keys))
        nn.init.normal_(self.selection_keys, std=1 / math.sqrt(self.dim))
        # Each emitted vector is a sum of books code vectors, so scale each
        # independent initial codebook accordingly. No output rescaling is used.
        nn.init.normal_(
            self.code_values,
            std=float(settings["value_initial_std"]) / math.sqrt(self.books),
        )

    @property
    def weight(self):
        """Expose original selector initialization without registering it twice."""
        return self.assignment_vectors.weight

    def assignment_logits(self, ids):
        query = F.normalize(self.assignment_vectors(ids), dim=-1, eps=1e-8)
        keys = F.normalize(self.selection_keys, dim=-1, eps=1e-8)
        return torch.einsum("nd,lcd->nlc", query, keys) / self.temperature

    def codes(self, ids):
        """Return deterministic learned codes; padding ID zero has codes -1."""
        codes = self.assignment_logits(ids).argmax(-1)
        return codes.masked_fill((ids == 0).unsqueeze(-1), -1)

    def encoded_items(self, ids, statistics=False):
        """Encode a vector of unique IDs, optionally returning balance metrics.

        The straight-through one-hot weights are exactly one-hot in the
        forward pass. In backward, the softmax supplies selector/key gradients;
        selected code values also receive the direct downstream MSE gradient.
        """
        logits = self.assignment_logits(ids)
        chosen = logits.argmax(-1)
        one_hot = F.one_hot(chosen, self.codes_per_book).to(logits.dtype)
        probabilities = None
        if self.training or statistics:
            probabilities = logits.softmax(-1)
        if self.training:
            weights = one_hot + (probabilities - probabilities.detach())
        else:
            weights = one_hot
        values = torch.einsum("nlc,lcd->nd", weights, self.code_values)
        nonpadding = ids != 0
        values = values * nonpadding.unsqueeze(-1)
        if not statistics:
            return values
        # Count each present source ID once, rather than weighting the code
        # regularizer by popularity or by repeated target ratings of a user.
        count = nonpadding.sum().clamp_min(1).to(values.dtype)
        marginal = (
            probabilities * nonpadding[:, None, None]
        ).sum(0) / count
        balance = (
            marginal
            * (marginal.clamp_min(1e-12).log() + math.log(self.codes_per_book))
        ).sum(-1).mean()
        hard_counts = (one_hot * nonpadding[:, None, None]).sum(0)
        usage = (hard_counts > 0).to(values.dtype).mean()
        return values, balance, usage

    def _pool(self, rows, per_sample_weights, statistics):
        if rows.ndim != 2 or rows.shape != per_sample_weights.shape:
            raise ValueError("Expected equally shaped [batch, ranks] IDs and weights")
        # An item is quantized once per batch even if it appears in many lists.
        # This avoids allocating [batch, K, books, codes] assignment tensors.
        unique, inverse = torch.unique(rows, sorted=True, return_inverse=True)
        encoded = self.encoded_items(unique, statistics=statistics)
        values = encoded[0] if statistics else encoded
        pooled = F.embedding_bag(
            inverse.reshape_as(rows), values,
            mode="sum", per_sample_weights=per_sample_weights,
        )
        if statistics:
            return pooled, encoded[1], encoded[2]
        return pooled

    def forward(self, rows, per_sample_weights):
        return self._pool(rows, per_sample_weights, statistics=False)

    def pool_with_statistics(self, rows, per_sample_weights):
        return self._pool(rows, per_sample_weights, statistics=True)


class SourceCodeResidual(ResidualList):
    """ResidualList with compositional source lookup and original rating head."""

    def __init__(self, lists, vocab, items, base, cfg):
        # Run the original constructor before creating any extra parameters:
        # encoder, target lookup and frozen bias are exactly the base LIST's
        # initialization under the same random seed.
        super().__init__(lists, vocab, items, base, cfg)
        self.source_code_settings = dict(SOURCE_CODE_DEFAULTS)
        self.source_code_settings.update(cfg.get("source_code", {}))
        # These added CPU initializations do not move the caller's RNG state
        # beyond the original model constructor (e.g. before its dropout draws).
        with torch.random.fork_rng(devices=[]):
            self.source_items = CompositionalSourceLookup(
                self.source_items.weight, self.source_code_settings
            )

    def training_loss(self, users, items, ratings):
        rows = self.lists[users]
        h, balance, usage = self.source_items.pool_with_statistics(
            rows, self.rank_weights.expand_as(rows)
        )
        z = self.encoder(h) * (rows[:, :1] != 0)
        predictions = self.base[items] + (z * self.target_items(items)).sum(-1)
        mse = (predictions - ratings).square().mean()
        total = mse + self.source_code_settings["balance_weight"] * balance
        return total, {
            "loss": total,
            "mse": mse,
            "code_balance": balance,
            "hard_usage_fraction": usage,
        }

    @torch.no_grad()
    def export_codes(self, chunk_size=4096):
        """Return CPU [vocab+1, books] codes in compact ID order; row0 is -1."""
        count = self.source_items.weight.shape[0]
        device = self.source_items.weight.device
        return torch.cat([
            self.source_items.codes(torch.arange(start, min(start + chunk_size, count), device=device)).cpu()
            for start in range(0, count, chunk_size)
        ])

    def parameter_report(self):
        return {
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "item_specific_selector_parameters": (
                self.source_items.assignment_vectors.weight.numel()
            ),
            "shared_selection_key_parameters": self.source_items.selection_keys.numel(),
            "shared_code_value_parameters": self.source_items.code_values.numel(),
            "source_output_contains_continuous_item_residual": False,
            "parameter_compression_claimed": False,
        }
