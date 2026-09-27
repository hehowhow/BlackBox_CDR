"""Only generalize the archived conditional Gaussian's rating-noise variance.

The conditional latent distribution, encoder, optimizer, prediction rule and
validation protocol are unchanged. observation_variance is a fixed scalar q,
not a standard deviation and not a newly learned network output.
"""
import math

from list_ambiguity_model import ConditionalList


def joint_nll_twice(mean, variance, a, b, c, counts, observation_variance):
    q = float(observation_variance)
    if not math.isfinite(q) or q <= 0:
        raise ValueError('observation_variance must be finite and positive')
    # Keep exactly the old arithmetic when q=1, including the loss scale.
    if q == 1.:
        return ConditionalList.gaussian_nll_twice(mean, variance, a, b, c)
    # Cov(r_u | list_u) = q I + V_u diag(var_u) V_u^T.
    # The archived Woodbury/determinant-lemma calculation works after scaling
    # rating sufficient statistics by q. Include n_u log(q); omit only the
    # same n_u log(2*pi) constant as the archived implementation.
    return (ConditionalList.gaussian_nll_twice(mean, variance, a / q, b / q, c / q)
            + counts * math.log(q))


class GaussianNoiseList(ConditionalList):
    def objective(self, users, items, residual, mask, sample_seed):
        assert self.family == 'gaussian', 'This sweep only changes gaussian_learned'
        q = self.cfg['observation_variance']
        if q == 1.:
            return super().objective(users, items, residual, mask, sample_seed)
        _, mean, variance = self.condition(users)
        a, b, c = self.rating_statistics(items, residual, mask)
        loss = joint_nll_twice(mean, variance, a, b, c, mask.sum(-1), q)
        return loss.sum() / mask.sum(), {}
