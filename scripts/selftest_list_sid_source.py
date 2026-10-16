"""CPU-only mathematical and gradient checks for LIST route A."""

import io
import json

import torch
from torch import nn

from list_sid_source_model import SourceCodeResidual
from small_task_comparison import ResidualList


def main():
    torch.set_num_threads(1)
    cfg = dict(dim=32, hidden=64, dropout=0.1)
    generator = torch.Generator().manual_seed(13)
    lists = torch.randint(1, 257, (16, 100), generator=generator)
    lists[-1].zero_()
    base = torch.linspace(2.0, 4.5, 19)
    torch.manual_seed(2020)
    original = ResidualList(lists, 256, len(base), base, cfg)
    original_rng_after_init = torch.random.get_rng_state().clone()
    torch.manual_seed(2020)
    model = SourceCodeResidual(lists, 256, len(base), base, cfg)
    assert torch.equal(torch.random.get_rng_state(), original_rng_after_init)
    for name, parameter in original.encoder.named_parameters():
        torch.testing.assert_close(parameter, dict(model.encoder.named_parameters())[name], rtol=0, atol=0)
    torch.testing.assert_close(original.target_items.weight, model.target_items.weight, rtol=0, atol=0)
    torch.testing.assert_close(original.source_items.weight, model.source_items.weight, rtol=0, atol=0)
    assert not model.item_bias.weight.requires_grad

    users = torch.arange(len(lists))
    items = users % len(base)
    ratings = torch.linspace(1.0, 5.0, len(users))
    model.eval()
    torch.testing.assert_close(model(users, items), base[items], rtol=0, atol=0)
    source = model.source_items
    ids = torch.arange(257)
    codes = model.export_codes()
    assert codes.shape == (257, 3) and (codes[0] == -1).all()
    encoded = source.encoded_items(ids)
    manual = sum(source.code_values[level, codes[1:, level]] for level in range(3))
    torch.testing.assert_close(encoded[1:], manual, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(encoded[0]) == 0
    actual_pool = source(lists, model.rank_weights.expand_as(lists))
    manual_pool = (encoded[lists] * model.rank_weights[None, :, None]).sum(1)
    torch.testing.assert_close(actual_pool, manual_pool, rtol=1e-5, atol=1e-7)

    # Give the unchanged zero-initialized final layer a nonzero state so the
    # target MSE can reach the source lookup. This is a test setup, not training
    # initialization; the preceding assertion verifies the actual epoch0 mean.
    with torch.no_grad():
        nn.init.normal_(model.encoder[-1].weight, std=0.1)
    model.train()
    loss, statistics = model.training_loss(users, items, ratings)
    assert torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    statistics["mse"].backward()  # No balance loss: rating task alone reaches codes.
    gradients = {}
    for name, parameter in [
        ("assignment_vectors", source.weight),
        ("selection_keys", source.selection_keys),
        ("code_values", source.code_values),
    ]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        gradients[name] = float(parameter.grad.abs().sum())
    assert source.weight.grad[0].abs().sum() == 0

    # Check hard assignments are learned, rather than a fixed initial hash.
    before = model.export_codes().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        mse = model.training_loss(users, items, ratings)[1]["mse"]
        mse.backward()
        optimizer.step()
    changed = int((model.export_codes()[1:] != before[1:]).any(-1).sum())
    assert changed > 0

    # No-list rows remain a true ordinary-mean fallback, and generate no code
    # balancing loss even when all network biases have changed during training.
    empty_users = torch.full((4,), len(lists) - 1, dtype=torch.long)
    empty_items = torch.arange(4)
    empty_loss, empty_stats = model.training_loss(empty_users, empty_items, ratings[:4])
    assert empty_stats["code_balance"] == 0
    assert empty_stats["hard_usage_fraction"] == 0
    torch.testing.assert_close(model.correction(empty_users, empty_items), torch.zeros(4), rtol=0, atol=0)
    assert torch.isfinite(empty_loss)

    # Train/eval both emit hard-code source vectors, and state_dict is enough
    # to restore every selector and shared code parameter exactly.
    train_encoded = source.encoded_items(ids).detach()
    model.eval()
    torch.testing.assert_close(source.encoded_items(ids), train_encoded, rtol=0, atol=0)
    prediction = model(users, items).detach()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = SourceCodeResidual(lists, 256, len(base), base, cfg)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(users, items), prediction, rtol=0, atol=0)
    torch.testing.assert_close(restored.export_codes(), model.export_codes(), rtol=0, atol=0)
    print(json.dumps({
        "status": "PASS", "device": "cpu", "common_initialization_identical": True,
        "initial_prediction_is_mean": True, "single_original_rank_pool": True,
        "hard_compositional_forward_train_and_eval": True,
        "target_mse_gradient_l1": gradients, "source_ids_changing_codes_from_mse": changed,
        "padding_and_no_list_inert": True, "state_reload_exact": True,
        "parameters": model.parameter_report(),
    }, indent=2))


if __name__ == "__main__":
    main()
