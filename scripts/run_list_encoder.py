"""One fixed-budget baby check: API lists -> target ratings, no distillation.

Reads only cached API output and target training data during learning. Test
ratings are opened only after the fixed final checkpoint has been saved.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(os.environ.get('CDR_RUN_ROOT', Path(__file__).resolve().parents[1]))
RUN_CONFIG = json.loads((ROOT / 'configs/first_round.json').read_text())
PAIR = RUN_CONFIG.get('src_tgt_pairs', {}).get(RUN_CONFIG['task'],
          dict(src='Books', tgt='Movies_and_TV'))
DATA = ROOT / 'data/ready/_8_2' / f"tgt_{PAIR['tgt']}_src_{PAIR['src']}"
CFG = dict(seed=RUN_CONFIG['seed'], epochs=10, dim=32, hidden=64, batch_size=2048,
           lr=0.001, weight_decay=0.00001, dropout=0.1, k=100,
           pooling='normalized 1/log2(rank+1)', scores=False,
           selection='fixed final epoch; no test-based selection',
           training='all target train ratings; no-list users update item parameters only')


class ListRecommender(nn.Module):
    def __init__(self, lists, item_count, mean):
        super().__init__()
        # Reconstruct from the existing API cache; do not duplicate this large
        # fixed lookup table in each checkpoint (especially for Books users).
        self.register_buffer('lists', lists, persistent=False)
        w = 1 / torch.log2(torch.arange(lists.shape[1], device=lists.device) + 2.)
        self.register_buffer('rank_weights', w / w.sum())
        self.source_items = nn.EmbeddingBag(int(lists.max()) + 1, CFG['dim'],
                                            mode='sum', padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(CFG['dim'], CFG['hidden']), nn.ReLU(),
                                     nn.Dropout(CFG['dropout']), nn.Linear(CFG['hidden'], CFG['dim']))
        self.target_items = nn.Embedding(item_count, CFG['dim'])
        self.item_bias = nn.Embedding(item_count, 1)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        nn.init.normal_(self.source_items.weight, std=0.1)
        with torch.no_grad():
            self.source_items.weight[0].zero_()
        nn.init.normal_(self.target_items.weight, std=0.1)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        lists = self.lists[users]
        h = self.source_items(lists, per_sample_weights=self.rank_weights.expand_as(lists))
        z = self.encoder(h) * (lists[:, :1] != 0)
        return self.mean + self.item_bias(items).squeeze(-1) + (z * self.target_items(items)).sum(-1)


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda')
    api = np.load(ROOT / 'queries/topk_scores.npz')
    api_users = api['user_ids'].astype(np.int64)
    # Compact opaque IDs without reading a source catalog or source parameters.
    vocabulary, compact = np.unique(api['item_ids'], return_inverse=True)
    compact = compact.reshape(api['item_ids'].shape) + 1
    assert compact.shape[1] == CFG['k']
    assert len(np.unique(api_users)) == len(api_users)
    train = pd.read_csv(DATA / 'train_tgt.csv', header=None).to_numpy()
    user_count = max(int(train[:, 0].max()), int(api_users.max())) + 1
    item_count = int(train[:, 1].max()) + 1
    lists = torch.zeros((user_count, CFG['k']), dtype=torch.long, device=device)
    lists[torch.as_tensor(api_users, device=device)] = torch.as_tensor(compact, device=device)
    users = torch.tensor(train[:, 0], dtype=torch.long, device=device)
    items = torch.tensor(train[:, 1], dtype=torch.long, device=device)
    labels = torch.tensor(train[:, 2], dtype=torch.float32, device=device)
    model = ListRecommender(lists, item_count, labels.mean().item()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    history = []
    print('CONFIG', json.dumps(CFG), flush=True)
    print('DATA', json.dumps(dict(train_ratings=len(train), api_users=len(api_users),
                                observed_source_items=len(vocabulary), target_items=item_count)), flush=True)
    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        epoch_start = time.monotonic()
        total = torch.zeros((), device=device)
        for idx in torch.randperm(len(users), device=device).split(CFG['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            pred = model(users[idx], items[idx])
            loss = (pred - labels[idx]).square().mean()
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(idx)
        row = dict(epoch=epoch, train_mse=total.item() / len(users),
                   seconds=time.monotonic() - epoch_start)
        assert np.isfinite(row['train_mse']), row
        history.append(row)
        (ROOT / 'results/list_encoder_training.json').write_text(json.dumps(history, indent=2))
        print('EPOCH', json.dumps(row), flush=True)
    checkpoint = ROOT / 'checkpoints/list_encoder.pt'
    torch.save(dict(config=CFG, state_dict=model.state_dict(), source_vocabulary=vocabulary.tolist()), checkpoint)

    # Final evaluation only; no checkpoint or hyperparameter selection on test.
    test = pd.read_csv(DATA / 'test.csv', header=None).to_numpy()
    assert not np.intersect1d(train[:, 0], test[:, 0]).size, 'Cold-user leakage'
    assert np.isin(test[:, 0], api_users).all(), 'Missing cold-user API lists'
    assert test[:, 1].max() < item_count, 'Unseen target item ID'
    tu = torch.tensor(test[:, 0], dtype=torch.long, device=device)
    ti = torch.tensor(test[:, 1], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(tu[s:s+4096], ti[s:s+4096])
                          for s in range(0, len(test), 4096)]).cpu().numpy()
    assert np.isfinite(pred).all()
    # Confirm exact sample alignment with the already completed WB reference.
    if (ROOT / 'results/wb_predictions.npz').exists():
        wb = np.load(ROOT / 'results/wb_predictions.npz')
        for key, column in [('uid', 0), ('iid', 1), ('y', 2)]:
            np.testing.assert_array_equal(wb[key], test[:, column])
    np.savez_compressed(ROOT / 'results/list_encoder_predictions.npz',
                        uid=test[:, 0], iid=test[:, 1], y=test[:, 2], prediction=pred)
    error = pred - test[:, 2]
    references = (json.loads((ROOT / 'results/summary.json').read_text())['mae']
                  if (ROOT / 'results/summary.json').exists() else {})
    result = dict(config=CFG, mae=float(np.abs(error).mean()),
                  rmse=float(np.sqrt(np.square(error).mean())),
                  references_mae=references, test_users=len(np.unique(test[:, 0])),
                  test_ratings=len(test), seconds=time.monotonic() - start,
                  scope='single fixed run; different architecture and target training objective, not a controlled mechanism study')
    if references:
        result['gap_to_wb'] = result['mae'] - references['wb']
        result['improvement_over_bb'] = references['bb'] - result['mae']
    (ROOT / 'results/list_encoder_summary.json').write_text(json.dumps(result, indent=2))
    print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
