"""Expose independent train/inference MALA controls without changing the model.

The archived objective and inference implementation are called directly. Only
their sampler controls are locally overridden, then restored in finally. Each
model is used synchronously in its own training process, never concurrently.
No parameters, layers, losses, sampler equations or random draws are added.
"""
from contextlib import contextmanager

from list_ambiguity_model import ConditionalList


class SplitSamplerList(ConditionalList):
    def __init__(self, lists, vocab, items, base, cfg, variant):
        super().__init__(lists, vocab, items, base, dict(cfg), variant)
        assert self.family == 'energy'
        assert cfg['hidden'] == 64 and cfg['dim'] == 32
        assert cfg['observation_variance'] == 1.
        for phase in ['train', 'eval']:
            assert cfg[f'{phase}_steps'] > 0
            assert isinstance(cfg[f'{phase}_steps'], int)
            assert cfg[f'{phase}_mala_step'] > 0
            assert cfg[f'{phase}_chains'] >= 2 and cfg[f'{phase}_chains'] % 2 == 0
        assert cfg['energy_bound'] > 0

    @contextmanager
    def _sampling_phase(self, phase):
        previous_cfg, previous_steps = self.cfg, self.steps
        self.cfg = dict(previous_cfg, mala_step=previous_cfg[f'{phase}_mala_step'])
        self.steps = previous_cfg[f'{phase}_steps']
        try:
            yield
        finally:
            self.cfg, self.steps = previous_cfg, previous_steps

    def objective(self, users, items, residual, mask, sample_seed):
        with self._sampling_phase('train'):
            return super().objective(users, items, residual, mask, sample_seed)

    def user_mean(self, users, sample_seed):
        with self._sampling_phase('eval'):
            return super().user_mean(users, sample_seed)
