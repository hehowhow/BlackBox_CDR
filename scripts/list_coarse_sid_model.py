"""The same LIST predictor for raw-ID and pre-released coarse-token lists.

This module never tokenizes source items. ``lists`` is the complete input
interface: either compact raw IDs for the reference condition or already
published, many-to-one tokens produced at the trusted boundary. Repeated
tokens retain their original rank positions. No ID-to-code map, source item
features, teacher parameters, or source interaction records are accepted.

The two families inherit the archived grouped-user LIST implementation. The
Gaussian uses its exact joint marginal likelihood over a user's ratings;
point uses squared error with the same grouped batches. Both predict from the
conditional mean and the ordinary training-item mean. We retain the unused,
frozen energy network to preserve the archived construction RNG sequence.
"""
import math

import torch

from list_ambiguity_model import ConditionalList


FIXED_DEFAULTS = {
    'dim': 32,
    'hidden': 64,
    'dropout': .1,
    'latent_std': .5,
    'observation_variance': 1.,
    'logvar_bounds': [math.log(.05 ** 2), math.log(1. ** 2)],
}
FAMILIES = {
    'point': 'point_matched',
    'point_matched': 'point_matched',
    'gaussian': 'gaussian_learned',
    'gaussian_learned': 'gaussian_learned',
}


def fixed_config(cfg):
    """Fill fixed study settings and reject accidental architecture searches."""
    effective = dict(cfg)
    for name, value in FIXED_DEFAULTS.items():
        if name in effective:
            supplied = effective[name]
            if isinstance(value, list):
                valid = len(supplied) == len(value) and all(
                    math.isclose(float(a), b, rel_tol=0., abs_tol=1e-12)
                    for a, b in zip(supplied, value))
            else:
                valid = math.isclose(float(supplied), value, rel_tol=0., abs_tol=1e-12)
            if not valid:
                raise ValueError(f'This fixed study requires {name}={value}; got {supplied}')
        effective[name] = value.copy() if isinstance(value, list) else value
    return effective


class PublishedTokenList(ConditionalList):
    """ConditionalList with a validated, already-published token interface."""

    def __init__(self, lists, vocab, items, base, cfg, family):
        if family not in FAMILIES:
            raise ValueError(f'Unsupported family: {family}')
        if lists.dtype != torch.long or lists.ndim != 2 or lists.shape[1] == 0:
            raise ValueError('lists must be a nonempty-width rank-2 long tensor')
        if vocab < 1 or items < 1:
            raise ValueError('vocab and items must be positive')
        if lists.numel() and (lists.min() < 0 or lists.max() > vocab):
            raise ValueError('Published token outside [0, vocab]')
        # The archived zero-list mask reads column zero. Require complete
        # source lists or all-zero rows so that mask is always meaningful.
        if not torch.all((lists == 0).all(1) | (lists > 0).all(1)):
            raise ValueError('A user row must contain a complete list or only padding')
        if tuple(base.shape) != (items,) or not base.is_floating_point():
            raise ValueError('base must contain one floating-point mean per target item')
        super().__init__(lists, vocab, items, base, fixed_config(cfg), FAMILIES[family])


def make_model(lists, vocab, items, base, cfg, family):
    """Build the point or learned-Gaussian arm; caller controls the RNG seed.

    Resetting the same seed for both families with the same token vocabulary
    produces exactly the same initial state for every parameter, including
    the frozen ones. The runner is responsible for any shared initialization
    between raw and coarse conditions whose vocabulary sizes differ.
    """
    return PublishedTokenList(lists, vocab, items, base, cfg, family)
