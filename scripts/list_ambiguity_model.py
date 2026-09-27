"""Conditional latent rating models; no source access beyond the LIST ID table.

The Gaussian model integrates ONE shared 32-D latent per training user over all
of that user's observed target ratings. The EBM replaces this Gaussian prior
with a bounded conditional energy tilt and trains by prior/posterior MCMC.
No KL bottleneck, reconstructed source histories, or target-conditioned test
sampler is used. Sampling is preconditioned Metropolis-adjusted Langevin.
"""
import math

import torch
from torch import nn

from small_task_comparison import ResidualList


VARIANTS = {
    'point_matched': {'family': 'point', 'steps': 0},
    'gaussian_fixed': {'family': 'gaussian_fixed', 'steps': 0},
    'gaussian_learned': {'family': 'gaussian', 'steps': 0},
    'energy_20': {'family': 'energy', 'steps': 20},
    'energy_60': {'family': 'energy', 'steps': 60},
}


def antithetic_normal(shape, device, generator, dtype):
    """Pairs reduce MC error; each chain still has the correct noise marginal."""
    assert shape[1] % 2 == 0
    x = torch.randn((shape[0], shape[1] // 2, shape[2]), device=device,
                    generator=generator, dtype=dtype)
    return torch.cat((x, -x), dim=1)


def langevin(energy_fn, condition, location, chol, steps, chains, step_size, seed):
    """MALA in whitened coordinates, using the Gaussian part as preconditioner.

    Density in t: exp(-||t||^2/2 - energy_fn(location + chol @ t, h)).
    Neither sampling trajectories nor accept decisions are differentiated.
    A Metropolis correction avoids treating discretized ULA as exact sampling.
    Short chains still need not mix fully. Prior/test callers pass no ratings.
    """
    location, chol, condition = location.detach(), chol.detach(), condition.detach()
    generator = torch.Generator(device=location.device).manual_seed(seed)
    shape = (len(location), chains, location.shape[-1])
    t = antithetic_normal(shape, location.device, generator, location.dtype)

    def value_grad(x):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            z = location[:, None, :] + torch.einsum('bij,bmj->bmi', chol, x)
            value = .5 * x.square().sum(-1) + energy_fn(z, condition)
            grad, = torch.autograd.grad(value.sum(), x)
        return value.detach(), grad.detach()

    value, grad = value_grad(t)
    accept_total = t.new_zeros(())
    for _ in range(steps):
        noise = antithetic_normal(shape, t.device, generator, t.dtype)
        proposal = t - .5 * step_size * grad + math.sqrt(step_size) * noise
        next_value, next_grad = value_grad(proposal)
        forward = (proposal - t + .5 * step_size * grad).square().sum(-1)
        reverse = (t - proposal + .5 * step_size * next_grad).square().sum(-1)
        log_accept = value - next_value + (forward - reverse) / (2 * step_size)
        uniform = torch.rand((shape[0], chains // 2), device=t.device,
                             generator=generator, dtype=t.dtype)
        log_uniform = torch.cat((uniform, uniform), dim=1).clamp_min(1e-30).log()
        accepted = log_uniform < log_accept.clamp_max(0)
        t = torch.where(accepted[..., None], proposal, t)
        value = torch.where(accepted, next_value, value)
        grad = torch.where(accepted[..., None], next_grad, grad)
        accept_total += accepted.float().mean()
    z = location[:, None, :] + torch.einsum('bij,bmj->bmi', chol, t)
    assert torch.isfinite(z).all(), 'Nonfinite Langevin sample'
    return z.detach(), accept_total / max(steps, 1)


class ConditionalList(ResidualList):
    def __init__(self, lists, vocab, items, base, cfg, variant):
        super().__init__(lists, vocab, items, base, cfg)
        self.cfg = cfg
        self.variant = variant
        self.family = VARIANTS[variant]['family']
        self.steps = VARIANTS[variant]['steps']
        # Construct after all old parameters so their initialization is identical.
        self.log_variance = nn.Linear(cfg['hidden'], cfg['dim'])
        nn.init.zeros_(self.log_variance.weight)
        nn.init.constant_(self.log_variance.bias, math.log(cfg['latent_std'] ** 2))
        self.energy_net = nn.Sequential(nn.Linear(2 * cfg['dim'], cfg['hidden']),
                                        nn.Tanh(), nn.Linear(cfg['hidden'], 1))
        nn.init.zeros_(self.energy_net[-1].weight)
        nn.init.zeros_(self.energy_net[-1].bias)
        self.log_variance.requires_grad_(self.family in {'gaussian', 'energy'})
        self.energy_net.requires_grad_(self.family == 'energy')

    def condition(self, users):
        rows = self.lists[users]
        h = self.source_items(rows, per_sample_weights=self.rank_weights.expand_as(rows))
        hidden = self.encoder[2](self.encoder[1](self.encoder[0](h)))
        mean = self.encoder[3](hidden)
        logvar = self.log_variance(hidden).clamp(*self.cfg['logvar_bounds'])
        return h, mean, logvar.exp()

    def tilt(self, z, h):
        h = h[:, None, :].expand(z.shape[0], z.shape[1], h.shape[-1])
        raw = self.energy_net(torch.cat((z, h), dim=-1)).squeeze(-1)
        bound = self.cfg['energy_bound']
        # A bounded tilt of a proper Gaussian is always normalizable.
        return bound * torch.tanh(raw / bound)

    def prior_energy(self, z, h, mean, variance):
        return (.5 * ((z - mean[:, None]) ** 2 / variance[:, None]).sum(-1)
                + .5 * variance.log().sum(-1)[:, None] + self.tilt(z, h))

    def rating_statistics(self, items, residual, mask):
        v = self.target_items(items) * mask[..., None]
        a = v.transpose(1, 2) @ v
        b = (v * residual[..., None]).sum(1)
        c = residual.square().sum(1)
        return a, b, c

    @staticmethod
    def likelihood_twice(z, a, b, c):
        return (torch.einsum('bmd,bdk,bmk->bm', z, a, z)
                - 2 * (z * b[:, None]).sum(-1) + c[:, None])

    @staticmethod
    def gaussian_nll_twice(mean, variance, a, b, c):
        """Exact joint marginal NLL, excluding constant n*log(2*pi), sigma=1."""
        scale = variance.sqrt()
        identity = torch.eye(mean.shape[-1], device=mean.device, dtype=mean.dtype)
        precision_white = identity + scale[:, :, None] * a * scale[:, None, :]
        chol = torch.linalg.cholesky(precision_white)
        residual_b = b - (a @ mean[..., None]).squeeze(-1)
        rhs = (scale * residual_b)[..., None]
        solved = torch.cholesky_solve(rhs, chol)
        point_sse = c - 2 * (mean * b).sum(-1) + torch.einsum('bi,bij,bj->b', mean, a, mean)
        return point_sse - (rhs * solved).sum((-2, -1)) + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def objective(self, users, items, residual, mask, sample_seed):
        h, mean, variance = self.condition(users)
        a, b, c = self.rating_statistics(items, residual, mask)
        count = mask.sum()
        if self.family == 'point':
            return self.likelihood_twice(mean[:, None], a, b, c).sum() / count, {}
        if self.family.startswith('gaussian'):
            return self.gaussian_nll_twice(mean, variance, a, b, c).sum() / count, {}

        with torch.no_grad():
            precision = a.detach() + torch.diag_embed(1 / variance.detach())
            chol_precision = torch.linalg.cholesky(precision)
            covariance = torch.cholesky_inverse(chol_precision)
            post_location = torch.cholesky_solve(
                (b.detach() + mean.detach() / variance.detach())[..., None], chol_precision).squeeze(-1)
            post_chol = torch.linalg.cholesky(covariance)
            prior_chol = torch.diag_embed(variance.detach().sqrt())
        # Coupled random draws lower variance without changing either marginal.
        prior, prior_accept = langevin(self.tilt, h, mean, prior_chol, self.steps,
                                      self.cfg['train_chains'], self.cfg['mala_step'], sample_seed)
        posterior, post_accept = langevin(self.tilt, h, post_location, post_chol, self.steps,
                                         self.cfg['train_chains'], self.cfg['mala_step'], sample_seed)
        # Correct maximum-likelihood gradient: E_post[dE+dL] - E_prior[dE].
        # This stochastic gradient surrogate is NOT a normalized reported NLL.
        positive = self.prior_energy(posterior, h, mean, variance)
        negative = self.prior_energy(prior, h, mean, variance)
        likelihood = .5 * self.likelihood_twice(posterior, a, b, c)
        objective = 2 * (positive.mean(1) - negative.mean(1) + likelihood.mean(1)).sum() / count
        return objective, {'prior_accept': prior_accept, 'posterior_accept': post_accept}

    def user_mean(self, users, sample_seed):
        # This inference API has no target rating or target-history argument.
        with torch.no_grad():
            h, mean, variance = self.condition(users)
        if self.family == 'energy':
            z, _ = langevin(self.tilt, h, mean, torch.diag_embed(variance.sqrt()), self.steps,
                            self.cfg['eval_chains'], self.cfg['mala_step'], sample_seed)
            mean = z.mean(1)
        return mean * (self.lists[users, :1] != 0)
