"""CPU equivalence and interface tests for the fixed published-token MALA arm."""
import inspect
import json
from unittest.mock import patch

import torch

import list_ambiguity_model as archived
from list_coarse_sid_energy_model import ENERGY_DEFAULTS, make_energy_model
from list_coarse_sid_model import make_model
from list_langevin_sensitivity_model import SplitSamplerList


def fixture():
    return (torch.tensor([[1, 2, 1, 3], [2, 1, 3, 1], [0, 0, 0, 0]]),
            torch.tensor([3.2, 4.1, 2.7]))


def compare_initialization():
    lists, base = fixture()
    torch.manual_seed(814)
    gaussian = make_model(lists, 3, 3, base, {}, 'gaussian')
    torch.manual_seed(814)
    energy = make_energy_model(lists, 3, 3, base, {})
    assert gaussian.state_dict().keys() == energy.state_dict().keys()
    for name, value in gaussian.state_dict().items():
        assert torch.equal(value, energy.state_dict()[name]), name
    gaussian_params = dict(gaussian.named_parameters())
    changed_trainability = []
    for name, parameter in energy.named_parameters():
        if parameter.requires_grad != gaussian_params[name].requires_grad:
            assert name.startswith('energy_net.'), name
            assert parameter.requires_grad and not gaussian_params[name].requires_grad
            changed_trainability.append(name)
    assert len(changed_trainability) == 4
    added = sum(p.numel() for name, p in energy.named_parameters()
                if name in changed_trainability)
    assert added == 4225
    energy.eval()
    users = torch.arange(3)
    torch.testing.assert_close(energy.user_mean(users, 1), torch.zeros(3, 32),
                               rtol=0., atol=1e-7)
    # Test Gaussian equivalence with a nonzero conditional mean, not merely at
    # the zero initialized last layer. Paired chains cancel for the zero tilt.
    with torch.no_grad():
        energy.encoder[-1].weight.normal_(std=.1)
        energy.encoder[-1].bias.fill_(.35)
    _, conditional_mean, _ = energy.condition(users)
    assert conditional_mean.abs().max() > .1
    expected = conditional_mean * (lists[:, :1] != 0)
    torch.testing.assert_close(energy.user_mean(users, 91), expected,
                               rtol=1e-6, atol=2e-7)
    return added


def compare_archived_nonzero_tilt():
    lists, base = fixture()
    torch.manual_seed(208)
    energy = make_energy_model(lists, 3, 3, base, {})
    with torch.no_grad():
        energy.encoder[-1].weight.normal_(std=.05)
        energy.encoder[-1].bias.normal_(std=.1)
        energy.log_variance.weight.normal_(std=.015)
        energy.energy_net[-1].weight.normal_(std=.12)
        energy.energy_net[-1].bias.fill_(.4)
    split = SplitSamplerList(lists, 3, 3, base, energy.cfg, 'energy_20')
    old = archived.ConditionalList(lists, 3, 3, base, energy.cfg, 'energy_20')
    for reference in (split, old):
        reference.load_state_dict(energy.state_dict())
    users = torch.tensor([0, 1])
    items = torch.tensor([[0, 1, 2], [1, 2, 0]])
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    residual = torch.tensor([[.9, -.4, .5], [-.2, .7, 0.]])
    all_users = torch.arange(3)
    results = []
    for model in (energy, split, old):
        model.train()
        torch.manual_seed(78)  # Identical training-dropout masks.
        loss, statistics = model.objective(users, items, residual, mask, 633)
        loss.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()
                     if p.grad is not None}
        model.eval()
        prediction = model.user_mean(all_users, 735)
        assert torch.equal(prediction[2], torch.zeros(32))
        h, mean, variance = model.condition(all_users)
        assert model.tilt(mean[:, None], h).abs().sum() > 0
        # Direct call into the archived sampler: verifies the inference wrapper
        # uses the same conditioning, Cholesky factor, chains and control values.
        draws, acceptance = archived.langevin(
            model.tilt, h, mean, torch.diag_embed(variance.sqrt()),
            20, 128, .25, 735)
        direct_prediction = draws.mean(1) * (lists[:, :1] != 0)
        assert torch.equal(prediction, direct_prediction)
        assert 0 < acceptance <= 1
        results.append((loss.detach(), statistics, gradients, prediction))
    expected = results[0]
    for observed in results[1:]:
        assert torch.equal(expected[0], observed[0])
        for name in expected[1]:
            assert torch.equal(expected[1][name], observed[1][name]), name
        assert expected[2].keys() == observed[2].keys()
        for name in expected[2]:
            assert torch.equal(expected[2][name], observed[2][name]), name
        assert torch.equal(expected[3], observed[3])
    for name, grad in expected[2].items():
        assert torch.isfinite(grad).all(), name
    assert expected[2]['energy_net.2.weight'].abs().sum() > 0
    return float(expected[0])


def check_sampler_controls_and_boundary():
    lists, base = fixture()
    model = make_energy_model(lists, 3, 3, base, {}).eval()
    users = torch.tensor([0, 1])
    items = torch.tensor([[0, 1], [1, 2]])
    residual = torch.tensor([[1., -.2], [.3, -.4]])
    mask = torch.ones_like(residual)
    # Test-only overrides establish that the train/eval controls are independent.
    # Production construction above and below accepts only the fixed defaults.
    model.cfg = dict(model.cfg, train_steps=3, eval_steps=5,
                     train_mala_step=.07, eval_mala_step=.19,
                     train_chains=4, eval_chains=6)
    saved_cfg, saved_steps = model.cfg, model.steps
    calls = []
    original_sampler = archived.langevin

    def observed_sampler(*args, **kwargs):
        calls.append((args[4], args[5], args[6]))
        return original_sampler(*args, **kwargs)

    with patch.object(archived, 'langevin', side_effect=observed_sampler):
        model.objective(users, items, residual, mask, 98)
        assert model.cfg is saved_cfg and model.steps == saved_steps
        model.user_mean(users, 99)
        assert model.cfg is saved_cfg and model.steps == saved_steps
    assert calls == [(3, 4, .07), (3, 4, .07), (5, 6, .19)], calls
    # Context manager restoration also holds when the inherited operation fails.
    for method, args in (('objective', (users, items, residual, mask, 7)),
                         ('user_mean', (users, 7))):
        with patch.object(archived.ConditionalList, method, side_effect=RuntimeError('test failure')):
            try:
                getattr(model, method)(*args)
            except RuntimeError as error:
                assert str(error) == 'test failure'
            else:
                raise AssertionError('Expected test exception')
        assert model.cfg is saved_cfg and model.steps == saved_steps
    assert list(inspect.signature(model.user_mean).parameters) == ['users', 'sample_seed']
    prediction = model.user_mean(users, 713)
    model.objective(users, items, residual * 7., mask, 714)
    assert torch.equal(prediction, model.user_mean(users, 713))
    # Retain every occurrence and its rank, including repeated coarse tokens.
    h, _, _ = model.condition(torch.arange(3))
    direct_pool = (model.source_items.weight[lists]
                   * model.rank_weights[None, :, None]).sum(1)
    torch.testing.assert_close(h, direct_pool, rtol=1e-6, atol=1e-7)
    with patch('builtins.open', side_effect=AssertionError('Unexpected file read')), \
            patch('io.open', side_effect=AssertionError('Unexpected file read')):
        checked = make_energy_model(lists, 3, 3, base, {}).eval()
        checked.user_mean(users, 713)
    for name, value in ENERGY_DEFAULTS.items():
        try:
            make_energy_model(lists, 3, 3, base, {name: value * 2})
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted changed production configuration: {name}')
    for bad_lists in (torch.tensor([[1, 0, 1, 2]]), torch.tensor([[1, 4, 1, 2]]),
                      torch.tensor([[1., 2., 1., 2.]])):
        try:
            make_energy_model(bad_lists, 3, 3, base, {})
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted invalid published token list')


def main():
    torch.set_num_threads(2)
    added = compare_initialization()
    loss = compare_archived_nonzero_tilt()
    check_sampler_controls_and_boundary()
    result = dict(status='PASS', gaussian_initial_state_bitwise_equal=True,
                  newly_trainable_energy_parameters=added, zero_tilt_mean_matches_gaussian=True,
                  nonzero_tilt_objective_gradients_and_sampler_match_archived_bitwise=True,
                  nonzero_tilt_test_objective=loss,
                  train_eval_sampling_controls_independent=True,
                  sampling_configuration_restored_even_after_exception=True,
                  prediction_has_no_target_rating_argument=True,
                  no_list_zero_prediction=True, duplicate_tokens_keep_original_ranks=True,
                  no_files_or_raw_id_map_read=True, fixed_parameters_enforced=True)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == '__main__':
    main()
