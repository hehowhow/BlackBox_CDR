"""Published-token LIST with the archived conditional energy/MALA model.

This is the energy extension of the frozen 9003 raw/coarse Gaussian study.
Only the previously allocated energy network becomes trainable: the parameter
state and RNG sequence at construction remain identical to its Gaussian arm.
The network has 4,225 parameters (64 -> 64 -> 1 including biases); neither the
32-dimensional latent nor the hidden width of 64 changes.

Input lists are already published tokens. No source-item features, original
item IDs behind coarse tokens, codebook maps, source histories or ratings are
accepted or loaded. The trusted publication step remains outside this model.
The exact archived objective and MALA equations are inherited without edits.
"""
import math

import torch

from list_coarse_sid_model import fixed_config
from list_langevin_sensitivity_model import SplitSamplerList


ENERGY_DEFAULTS = {
    'train_steps': 20,
    'eval_steps': 20,
    'train_mala_step': .25,
    'eval_mala_step': .25,
    'train_chains': 32,
    'eval_chains': 128,
    'energy_bound': 5.,
    # Kept for compatibility with the archived ConditionalList; phase-specific
    # controls override this locally and restore the original config afterward.
    'mala_step': .25,
}


def fixed_energy_config(cfg):
    """Apply the fixed 9003 Gaussian settings and the no-search MALA controls."""
    effective = fixed_config(cfg)
    for name, value in ENERGY_DEFAULTS.items():
        if name in effective:
            supplied = effective[name]
            if (isinstance(supplied, bool)
                    or not math.isclose(float(supplied), value, rel_tol=0., abs_tol=1e-12)):
                raise ValueError(f'This fixed study requires {name}={value}; got {supplied}')
            if name.endswith('_steps') or name.endswith('_chains'):
                if not isinstance(supplied, int):
                    raise ValueError(f'{name} must be an integer')
        effective[name] = value
    return effective


class PublishedTokenEnergyList(SplitSamplerList):
    """A validated published-token interface around the unchanged MALA model."""

    def __init__(self, lists, vocab, items, base, cfg):
        # Match PublishedTokenList's input boundary, without constructing a
        # temporary model (which would consume RNG and alter initialization).
        if lists.dtype != torch.long or lists.ndim != 2 or lists.shape[1] == 0:
            raise ValueError('lists must be a nonempty-width rank-2 long tensor')
        if vocab < 1 or items < 1:
            raise ValueError('vocab and items must be positive')
        if lists.numel() and (lists.min() < 0 or lists.max() > vocab):
            raise ValueError('Published token outside [0, vocab]')
        if not torch.all((lists == 0).all(1) | (lists > 0).all(1)):
            raise ValueError('A user row must contain a complete list or only padding')
        if tuple(base.shape) != (items,) or not base.is_floating_point():
            raise ValueError('base must contain one floating-point mean per target item')
        super().__init__(lists, vocab, items, base, fixed_energy_config(cfg), 'energy_20')


def make_energy_model(lists, vocab, items, base, cfg):
    """Construct the fixed 20-step arm; the runner controls initialization seed.

    With the same input vocabulary and RNG seed, every initial parameter and
    persistent buffer equals the 9003 Gaussian model. As in 9003, the runner
    aligns downstream initial parameters across different raw/coarse vocabulary
    sizes. Use ``user_mean`` for the energy model's sampled inference mean;
    the inherited residual ``forward`` does not perform MALA sampling.
    """
    return PublishedTokenEnergyList(lists, vocab, items, base, cfg)
