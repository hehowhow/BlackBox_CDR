"""Small GPU correctness checks before launching the preregistered experiment."""
import numpy as np
import torch
from residual_experiment import Residual, SourceMF, seed_all, state_hash

cfg = dict(student_dim=4, student_hidden=8, student_dropout=.1, k=3)
values = torch.tensor([3., 4., 2.], device='cuda')
seen = torch.tensor([True, True, False], device='cuda')
lists = torch.tensor([[1, 2, 3], [3, 2, 1]], device='cuda')
x = torch.tensor([[0, 0], [1, 1], [0, 2], [3, 0]], device='cuda')
seed_all(2020)
model = Residual(values, seen, lists, 3, cfg, 'real').cuda()
model.eval()
torch.testing.assert_close(model(x), values[x[:, 1]], rtol=0, atol=0)
initial = state_hash(model)
seed_all(2020)
other = Residual(values, seen, lists.flip(0), 3, cfg, 'shuffled').cuda()
assert state_hash(other) == initial
opt = torch.optim.Adam(model.parameters(), lr=.01)
for _ in range(3):
    model.train()
    opt.zero_grad()
    loss = (model(x)-torch.tensor([1., 2., 5., 1.], device='cuda')).square().mean()
    loss.backward()
    opt.step()
assert model.encoder[-1].weight.abs().sum().item() > 0
assert model.source_items.weight.grad.abs().sum().item() > 0
model.eval()
assert model(x)[2] == values[2]
assert not torch.equal(model(x), values[x[:, 1]])
source = SourceMF(4, 5, 3).cuda()
z = torch.tensor([[0, 1], [3, 2]], device='cuda')
torch.testing.assert_close(source(z), (source.users.weight[z[:, 0]]*source.items.weight[z[:, 1]]).sum(1))
print('SELFTEST_PASS: zero residual, paired initialization, trainable gradients, cold-item fallback, MF forward')
