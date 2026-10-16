"""CPU numerical checks for the published-token 2x2 LIST experiment."""
import inspect
import json
import math
from unittest.mock import patch

import torch

from list_ambiguity_model import ConditionalList
from list_coarse_sid_model import make_model
from small_task_comparison import ResidualList


def test_joint_marginal():
    """Compare values and all active gradients with a dense Gaussian integral."""
    generator = torch.Generator().manual_seed(91)
    dtype = torch.float64
    # Users have different numbers of ratings; masked ratings are excluded.
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=dtype)
    mean = torch.randn(2, 3, generator=generator, dtype=dtype).requires_grad_()
    logvar = torch.randn(2, 3, generator=generator, dtype=dtype).requires_grad_()
    values = torch.randn(2, 5, 3, generator=generator, dtype=dtype).requires_grad_()
    ratings = torch.randn(2, 5, generator=generator, dtype=dtype).requires_grad_()
    variance = logvar.exp()
    v = values * mask[..., None]
    r = ratings * mask
    a = v.transpose(1, 2) @ v
    b = (v * r[..., None]).sum(1)
    c = r.square().sum(1)
    actual = ConditionalList.gaussian_nll_twice(mean, variance, a, b, c)
    expected = []
    for user in range(2):
        count = int(mask[user].sum())
        vu = values[user, :count]
        covariance = torch.eye(count, dtype=dtype) + (vu * variance[user]) @ vu.T
        distribution = torch.distributions.MultivariateNormal(
            vu @ mean[user], covariance_matrix=covariance)
        expected.append(-2 * distribution.log_prob(ratings[user, :count])
                        - count * math.log(2 * math.pi))
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    parameters = (mean, logvar, values, ratings)
    actual_grad = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), parameters)
    for observed, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(observed, reference, atol=1e-9, rtol=1e-9)
    return {'max_value_error': float((actual - expected).detach().abs().max()),
            'max_gradient_error': max(float((a - b).abs().max())
                                      for a, b in zip(actual_grad, expected_grad))}


def test_model_interface():
    # Codes repeat because different raw source items may share a coarse code.
    lists = torch.tensor([[1, 2, 1, 3], [2, 1, 3, 1], [0, 0, 0, 0]])
    base = torch.tensor([3.2, 4.1, 2.7])
    cfg = {}
    torch.manual_seed(814)
    point = make_model(lists, 3, 3, base, cfg, 'point')
    torch.manual_seed(814)
    gaussian = make_model(lists, 3, 3, base, cfg, 'gaussian')
    for key, value in point.state_dict().items():
        assert torch.equal(value, gaussian.state_dict()[key]), key
    torch.manual_seed(814)
    original = ResidualList(lists, 3, 3, base, point.cfg)
    for key, value in original.state_dict().items():
        assert torch.equal(value, point.state_dict()[key]), key
    users = torch.arange(3)
    targets = torch.tensor([0, 1, 2])
    for model in (point, gaussian):
        model.eval()
        assert torch.equal(model.user_mean(users, 5), torch.zeros(3, 32))
        assert torch.equal(model(users, targets), base[targets])
        assert torch.equal(model.lists, lists)
        assert all(not p.requires_grad for p in model.energy_net.parameters())
        with torch.no_grad():
            model.encoder[-1].weight.normal_(std=.05)
            model.encoder[-1].bias.fill_(.7)
        h, _, _ = model.condition(users)
        expected = (model.source_items.weight[lists]
                    * model.rank_weights[None, :, None]).sum(1)
        torch.testing.assert_close(h, expected, rtol=1e-6, atol=1e-7)
        # Repeated code 1 contributes at both original ranks, not once.
        expected_first = (model.source_items.weight[1]
                          * (model.rank_weights[0] + model.rank_weights[2])
                          + model.source_items.weight[2] * model.rank_weights[1]
                          + model.source_items.weight[3] * model.rank_weights[3])
        torch.testing.assert_close(h[0], expected_first, rtol=1e-6, atol=1e-7)
        assert not torch.equal(h[0], h[1]), 'Rank order was discarded'
        assert torch.equal(model.user_mean(users, 7)[2], torch.zeros(32))
        assert model(users, targets)[2] == base[2]
        item_matrix = torch.tensor([[0, 1, 2], [1, 2, 0]])
        mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
        residual = torch.tensor([[1., -.3, .5], [-.2, 1., 0.]])
        prediction_before = model.user_mean(users, 27).clone()
        loss, statistics = model.objective(users[:2], item_matrix, residual, mask, 123)
        assert statistics == {} and torch.isfinite(loss)
        if model.family == 'point':
            _, mean, _ = model.condition(users[:2])
            predictions = (model.target_items(item_matrix) * mean[:, None]).sum(-1)
            direct_mse = ((predictions - residual).square() * mask).sum() / mask.sum()
            torch.testing.assert_close(loss, direct_mse, rtol=1e-5, atol=1e-7)
        # Target labels belong to the training objective, never inference.
        changed_loss, _ = model.objective(users[:2], item_matrix,
                                         residual * 3., mask, 456)
        assert changed_loss != loss
        assert torch.equal(prediction_before, model.user_mean(users, 456))
        assert list(inspect.signature(model.user_mean).parameters) == ['users', 'sample_seed']
        loss.backward()
        for module in (model.source_items, model.target_items, model.encoder[-1]):
            assert module.weight.grad is not None
            assert torch.isfinite(module.weight.grad).all()
            assert module.weight.grad.abs().sum() > 0
        if model.family == 'gaussian':
            assert model.log_variance.weight.grad.abs().sum() > 0
            assert model.log_variance.bias.grad.abs().sum() > 0
        else:
            assert all(not p.requires_grad for p in model.log_variance.parameters())
    # Once imported, model construction/inference require no raw IDs or files.
    with patch('builtins.open', side_effect=AssertionError('Unexpected file read')), \
            patch('io.open', side_effect=AssertionError('Unexpected file read')):
        for family in ('point_matched', 'gaussian_learned'):
            model = make_model(lists, 3, 3, base, cfg, family).eval()
            model.user_mean(users, 91)
            model(users, targets)
    # Invalid configurations fail instead of silently changing the experiment.
    for bad in ({'observation_variance': 2.}, {'latent_std': .2},
                {'hidden': 128}, {'logvar_bounds': [-2., 1.]}):
        try:
            make_model(lists, 3, 3, base, bad, 'point')
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted inconsistent settings: {bad}')


def main():
    torch.set_num_threads(2)
    numeric = test_joint_marginal()
    test_model_interface()
    result = dict(status='PASS', exact_multi_rating_marginal=numeric,
                  shared_parameter_initialization=True,
                  archived_initialization_preserved=True,
                  repeated_tokens_and_original_ranks_preserved=True,
                  initial_zero_correction=True, no_list_zero_correction=True,
                  inference_independent_of_target_ratings=True,
                  model_reads_no_files_or_raw_id_maps=True,
                  active_module_gradients=True, fixed_config_enforced=True)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == '__main__':
    main()
