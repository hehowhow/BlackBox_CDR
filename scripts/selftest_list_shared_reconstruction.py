"""Check masked-ID isolation, rank sensitivity, gradients and MALA compatibility."""
import argparse
import copy
import io
import json
from pathlib import Path

import torch

from list_raw_fixed_ablation import FIXED
from list_raw_fixed_ablation_model import matched_pair
from list_shared_reconstruction_model import (
    PATTERN_DEFAULTS, make_shared_reconstruction_model, inherited_parameter_hash)
from list_ambiguity_model import ConditionalList


def run():
    torch.set_num_threads(1)
    cfg = dict(FIXED, k=8, shared_reconstruction=dict(PATTERN_DEFAULTS, slots=2))
    lists = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1],
                          [1, 3, 5, 7, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19, 20],
                          [0, 0, 0, 0, 0, 0, 0, 0]])
    fit_users = torch.tensor([0, 1, 2])
    base = torch.tensor([3., 3.5, 4., 2.5])
    model, record = make_shared_reconstruction_model(lists, 20, 4, 4, base, cfg, 2021, fit_users)
    baseline, _ = matched_pair(lists, 20, 4, 4, base, cfg, 2021, 'energy')
    model.eval(); baseline.eval()
    assert inherited_parameter_hash(model) == record['raw_initial_sha256']
    assert torch.equal(model.pattern_candidate_ids, torch.arange(1, 13))
    assert not any('frequen' in n or 'counts' in n or 'band' in n for n, _ in model.named_buffers())
    users = torch.arange(len(lists))
    parts = model.encode_source(users, True)
    torch.testing.assert_close(parts['h'], baseline.condition(users)[0], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(parts['reconstructed'] + parts['residual'], parts['original'], atol=1e-7, rtol=1e-5)
    assert not torch.allclose(parts['slots'][0], parts['slots'][1]), 'Rank permutation must affect shared slots'
    assert torch.count_nonzero(parts['h'][-1]) == 0
    masked = torch.zeros(1, 8, dtype=torch.bool); masked[:, [0, 3]] = True
    rows = lists[:1].clone()
    before = model.reconstruct(rows, masked)
    rows[:, 0] = 17; rows[:, 3] = 18
    after = model.reconstruct(rows, masked)
    for index in [1, 2, 3, 4]:
        torch.testing.assert_close(before[index], after[index], rtol=0, atol=0)
    rows[:, 1] = 19
    assert not torch.allclose(before[1], model.reconstruct(rows, masked)[1])
    assert torch.count_nonzero(before[3][:, :, masked[0]]) == 0
    loss, _ = model.masked_reconstruction_loss(fit_users, 83)
    assert torch.isfinite(loss)
    state = torch.get_rng_state().clone()
    repeat, _ = model.masked_reconstruction_loss(fit_users, 83)
    torch.testing.assert_close(loss, repeat, rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state()), 'Auxiliary masking must not consume main RNG'
    try:
        model.masked_reconstruction_loss(torch.tensor([3]), 83)
    except ValueError:
        pass
    else:
        raise AssertionError('Held-out user entered masked reconstruction training')
    assert model.prior_energy.__func__ is ConditionalList.prior_energy
    torch.testing.assert_close(model.user_mean(users, 73), baseline.user_mean(users, 73), atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        model.encoder[3].weight.normal_(std=.02)
        model.energy_net[-1].weight.normal_(std=.02)
        model.pattern_shared_query.normal_(std=.02)
        model.pattern_residual_offset.normal_(std=.02)
        model.pattern_residual_query.weight.normal_(std=.02)
    items = torch.tensor([[0, 1], [1, 2], [2, 3]])
    residuals = torch.tensor([[.7, -.4], [.3, -.8], [.2, .9]])
    model.train()
    loss, extra = model.objective(fit_users, items, residuals, torch.ones(3, 2), 81)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        gradients[name] = float(parameter.grad.abs().sum())
        if name.startswith('pattern_'):
            assert gradients[name] > 0, name
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.step(); model.eval()
    clone = copy.deepcopy(model)
    stream = io.BytesIO(); torch.save(model.state_dict(), stream); stream.seek(0)
    with torch.no_grad():
        clone.pattern_queries.add_(1)
    clone.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(model.user_mean(users, 87), clone.user_mean(users, 87), rtol=0, atol=0)
    return dict(status='PASS', no_frequency_statistics=True, training_ID_membership_only=True,
                masked_ID_no_input_path=True, decoder_rank_and_slots_only=True,
                rank_permutation_changes_shared_slots=True, held_out_auxiliary_training_blocked=True,
                reconstruction_plus_residual_equals_original=True,
                initial_condition_matches_rank_pooling=True, original_initial_parameters_match=True,
                original_initial_MALA_predictions_match=True, auxiliary_uses_separate_RNG=True,
                padding_fallback=True, inherited_energy_and_MALA=True,
                checkpoint_restoration_predictions=True, gradient_checks=gradients,
                extra_parameters=record['extra_trainable_parameters'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
