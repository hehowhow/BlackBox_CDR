"""Rank-aware shared-pattern reconstruction and residual attention for LIST.

No item counts, frequency bands, source features or coarse-code maps are used.
A small bank of learned queries compresses the ordered list. A decoder sees
only those slots and rank queries, never a direct item-ID skip connection.
The second attention reads item embeddings minus their decoded reconstruction.
Masked ID prediction trains the reconstruction on training-user lists only.
The conditional energy, Gaussian heads, MALA and target scorer are inherited.
"""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from list_coarse_sid_energy_model import PublishedTokenEnergyList
from list_raw_fixed_ablation_model import matched_pair


PATTERN_DEFAULTS = dict(slots=4, mask_rate=0.2, reconstruction_weight=0.1,
                        reconstruction_normalization='cross_entropy/log(training_ID_count)')
PATTERN_SEED_OFFSET = 71016
MASK_SEED_OFFSET = 91016


class SharedReconstructionEnergyList(PublishedTokenEnergyList):
    def __init__(self, lists, vocab, items, base, cfg, fit_users):
        super().__init__(lists, vocab, items, base, cfg)
        spec = cfg['shared_reconstruction']
        dim, k, slots = cfg['dim'], lists.shape[1], int(spec['slots'])
        if not 1 < slots < k or not 0 < spec['mask_rate'] < 1:
            raise ValueError('Require a genuine slot bottleneck and partial masking')
        fit_users = torch.as_tensor(fit_users, device=lists.device, dtype=torch.long)
        if not len(fit_users) or len(fit_users.unique()) != len(fit_users):
            raise ValueError('Require unique nonempty training users')
        if not (lists[fit_users] > 0).all():
            raise ValueError('Training users must have complete source lists')
        # Membership only. No counts or empirical-frequency weights are stored.
        candidates = lists[fit_users].unique(sorted=True)
        if len(candidates) < 2:
            raise ValueError('Masked prediction requires at least two training IDs')
        self.register_buffer('pattern_fit_users', fit_users, persistent=False)
        self.register_buffer('pattern_candidate_ids', candidates)
        self.pattern_rank = nn.Parameter(torch.empty(k, dim))
        self.pattern_queries = nn.Parameter(torch.empty(slots, dim))
        self.pattern_input_key = nn.Linear(dim, dim, bias=False)
        self.pattern_decoder_query = nn.Linear(dim, dim, bias=False)
        self.pattern_decoder_key = nn.Linear(dim, dim, bias=False)
        self.pattern_shared_key = nn.Linear(dim, dim, bias=False)
        self.pattern_shared_query = nn.Parameter(torch.zeros(dim))
        self.pattern_residual_key = nn.Linear(dim, dim, bias=False)
        self.pattern_residual_query = nn.Linear(dim, dim, bias=False)
        self.pattern_residual_offset = nn.Parameter(torch.zeros(dim))
        self.pattern_shared_output = nn.Linear(dim, dim, bias=False)
        self.pattern_residual_output = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.pattern_rank, std=0.02)
        nn.init.normal_(self.pattern_queries, std=0.1)
        for layer in [self.pattern_input_key, self.pattern_decoder_query,
                      self.pattern_decoder_key, self.pattern_shared_key,
                      self.pattern_residual_key]:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.pattern_residual_query.weight)
        nn.init.eye_(self.pattern_shared_output.weight)
        nn.init.eye_(self.pattern_residual_output.weight)

    def reconstruct(self, rows, masked=None):
        """Rank-only decoder; masked IDs have no path to slots or decoder."""
        embeddings = F.embedding(rows, self.source_items.weight, padding_idx=0)
        valid = rows > 0
        visible = valid if masked is None else valid & ~masked
        # Zero masked values as well as excluding their attention logits.
        inputs = embeddings * visible[..., None]
        # Normalize attention inputs so the small inherited ID initialization
        # does not make all learned slots start with indistinguishable weights.
        keys = self.pattern_input_key(F.layer_norm(inputs + self.pattern_rank[None], (self.cfg['dim'],)))
        scale = self.cfg['dim'] ** -0.5
        queries = F.layer_norm(self.pattern_queries, (self.cfg['dim'],))
        logits = torch.einsum('md,bkd->bmk', queries, keys) * scale
        logits = logits + self.rank_weights.log()[None, None]
        logits = logits.masked_fill(~visible[:, None], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1) * visible[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        slots = weights @ inputs
        queries = self.pattern_decoder_query(F.layer_norm(self.pattern_rank, (self.cfg['dim'],)))
        decoder_logits = torch.einsum('kd,bmd->bkm', queries,
                                     self.pattern_decoder_key(F.layer_norm(slots, (self.cfg['dim'],)))) * scale
        decoder_weights = torch.softmax(decoder_logits, dim=-1)
        reconstructed = (decoder_weights @ slots) * valid[..., None]
        return embeddings, reconstructed, slots, weights, decoder_weights

    def encode_source(self, users, return_parts=False):
        rows = self.lists[users]
        original, reconstructed, slots, slot_weights, decoder_weights = self.reconstruct(rows)
        residual = original - reconstructed
        scale, log_rank = self.cfg['dim'] ** -0.5, self.rank_weights.log()
        shared_logits = (self.pattern_shared_key(reconstructed + self.pattern_rank[None])
                         * self.pattern_shared_query).sum(-1) * scale
        shared_weights = torch.softmax(shared_logits + log_rank, dim=-1)
        shared = (shared_weights[..., None] * reconstructed).sum(1)
        query = self.pattern_residual_query(shared) + self.pattern_residual_offset
        residual_logits = (self.pattern_residual_key(residual + self.pattern_rank[None])
                           * query[:, None]).sum(-1) * scale
        residual_weights = torch.softmax(residual_logits + log_rank, dim=-1)
        specific = (residual_weights[..., None] * residual).sum(1)
        h = self.pattern_shared_output(shared) + self.pattern_residual_output(specific)
        if return_parts:
            return dict(h=h, shared=shared, residual_summary=specific, original=original,
                        reconstructed=reconstructed, residual=residual, slots=slots,
                        slot_weights=slot_weights, decoder_weights=decoder_weights,
                        shared_weights=shared_weights, residual_weights=residual_weights)
        return h

    def condition(self, users):
        h = self.encode_source(users)
        hidden = self.encoder[2](self.encoder[1](self.encoder[0](h)))
        mean = self.encoder[3](hidden)
        logvar = self.log_variance(hidden).clamp(*self.cfg['logvar_bounds'])
        return h, mean, logvar.exp()

    def masked_reconstruction_loss(self, users, sample_seed):
        if not torch.isin(users, self.pattern_fit_users).all():
            raise ValueError('Masked reconstruction may only train on fit users')
        rows = self.lists[users]
        count = min(rows.shape[1] - 1, max(1, round(rows.shape[1] * self.cfg['shared_reconstruction']['mask_rate'])))
        generator = torch.Generator(device=rows.device).manual_seed(sample_seed + MASK_SEED_OFFSET)
        positions = torch.rand(rows.shape, device=rows.device, generator=generator).argsort(-1)[:, :count]
        masked = torch.zeros_like(rows, dtype=torch.bool).scatter_(1, positions, True)
        _, reconstructed, _, _, _ = self.reconstruct(rows, masked)
        # Tied ID classifier; normalization only rescales the classifier input.
        prediction = F.layer_norm(reconstructed[masked], (self.cfg['dim'],))
        vocabulary = self.source_items.weight[self.pattern_candidate_ids]
        labels = torch.searchsorted(self.pattern_candidate_ids, rows[masked])
        assert torch.equal(self.pattern_candidate_ids[labels], rows[masked])
        logits = prediction @ vocabulary.T
        ce = F.cross_entropy(logits, labels)
        normalized = ce / math.log(len(self.pattern_candidate_ids))
        accuracy = (logits.argmax(-1) == labels).float().mean().detach()
        return normalized, dict(reconstruction_ce=ce.detach(), masked_accuracy=accuracy,
                                masked_positions=int(masked.sum()))

    def objective(self, users, items, residual, mask, sample_seed):
        original_loss, extra = super().objective(users, items, residual, mask, sample_seed)
        reconstruction, details = self.masked_reconstruction_loss(users, sample_seed)
        loss = original_loss + self.cfg['shared_reconstruction']['reconstruction_weight'] * reconstruction
        return loss, dict(extra, **details, recommendation_objective=original_loss.detach(),
                          reconstruction_normalized=reconstruction.detach())


def inherited_parameter_hash(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not name.startswith('pattern_'):
            h.update(parameter.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_shared_reconstruction_model(lists, vocab, coarse_vocab, items, base, cfg, seed, fit_users):
    """Preserve archived initial parameters and RNG; coarse_vocab is an RNG size only."""
    baseline, record = matched_pair(lists, vocab, coarse_vocab, items, base, cfg, seed, 'energy')
    devices = [lists.device.index or 0] if lists.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed + PATTERN_SEED_OFFSET)
        model = SharedReconstructionEnergyList(lists, vocab, items, base, cfg, fit_users).to(lists.device)
        loaded = model.load_state_dict(baseline.state_dict(), strict=False)
        assert not loaded.unexpected_keys
        assert all(k.startswith('pattern_') for k in loaded.missing_keys)
    assert inherited_parameter_hash(model) == record['raw_initial_sha256']
    extra = sum(p.numel() for n, p in model.named_parameters() if n.startswith('pattern_'))
    return model, dict(**record, extra_trainable_parameters=extra,
                       reconstruction_training_ids=len(model.pattern_candidate_ids))
