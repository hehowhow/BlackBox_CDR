"""Fresh source MF service and paired residual students; never read test labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from book_movie_protocol import BASE, digest, save


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False


def guard_test_labels(task, student=False):
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if (path.name == 'test.csv' or path.name == 'test_predictions.npz' or
            BASE/'data/mid' in path.parents or BASE/'data/raw' in path.parents or
            (student and path.name in {'source.pt', 'source.csv'})):
            raise RuntimeError(f'Forbidden label/privileged-source read: {path}')
    sys.addaudithook(hook)


def load_frame(task, name):
    path = task/f'data/{name}.csv'
    manifest = json.loads((task/'manifest.json').read_text())
    assert digest(path) == manifest['hashes'][f'data/{name}.csv']
    return pd.read_csv(path, header=None, names=['uid', 'iid', 'y'])


class SourceMF(nn.Module):
    def __init__(self, users, items, dim):
        super().__init__()
        self.users = nn.Embedding(users, dim)
        self.items = nn.Embedding(items, dim)

    def forward(self, x):
        return (self.users(x[:, 0])*self.items(x[:, 1])).sum(1)


def source(task, cfg, manifest, seed, out):
    guard_test_labels(task)
    assert not (out/'source.pt').exists()
    seed_all(seed)
    frame = load_frame(task, 'source')
    x = torch.tensor(frame[['uid', 'iid']].to_numpy(), dtype=torch.long, device='cuda')
    y = torch.tensor(frame.y.to_numpy(), dtype=torch.float32, device='cuda')
    model = SourceMF(manifest['source_users'], manifest['source_items'], cfg['source_dim']).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg['source_lr'])
    generator = torch.Generator(device='cuda').manual_seed(seed+100)
    history = []
    for epoch in range(1, cfg['source_epochs']+1):
        tick = time.monotonic()
        total = torch.zeros((), device='cuda')
        for idx in torch.randperm(len(x), device='cuda', generator=generator).split(cfg['source_batch']):
            opt.zero_grad(set_to_none=True)
            loss = (model(x[idx])-y[idx]).square().mean()
            loss.backward()
            opt.step()
            total += loss.detach()*len(idx)
        row = dict(epoch=epoch, mse=total.item()/len(x), seconds=time.monotonic()-tick)
        assert np.isfinite(row['mse'])
        history.append(row)
        save(out/'source_training.json', history)
        print('SOURCE_EPOCH', task.name, seed, json.dumps(row), flush=True)
    torch.save(model.state_dict(), out/'source.pt')
    ids = []
    with torch.no_grad():
        for start in range(0, manifest['overlap_users'], 128):
            score = model.users.weight[start:min(start+128, manifest['overlap_users'])] @ model.items.weight.T
            ids.append(score.topk(cfg['k'], dim=1).indices.cpu().numpy().astype(np.int32))
    np.savez_compressed(out/'topk.npz', user_ids=np.arange(manifest['overlap_users']), item_ids=np.concatenate(ids))
    save(out/'source_complete.json', dict(source_sha256=digest(out/'source.pt'), topk_sha256=digest(out/'topk.npz'),
                                        source_data_sha256=manifest['hashes']['data/source.csv'], seed=seed,
                                        epochs=cfg['source_epochs'], target_labels_used=False))
    print('SOURCE_COMPLETE', task.name, seed, flush=True)


class Residual(nn.Module):
    def __init__(self, values, seen, lists, vocabulary, cfg, arm):
        super().__init__()
        self.arm = arm
        self.register_buffer('baseline', values)
        self.register_buffer('seen', seen)
        self.register_buffer('lists', lists, persistent=False)
        self.bias = nn.Embedding(len(values), 1)
        nn.init.zeros_(self.bias.weight)
        if arm != 'target_bias':
            self.source_items = nn.EmbeddingBag(vocabulary+1, cfg['student_dim'], mode='sum', padding_idx=0)
            self.target_items = nn.Embedding(len(values), cfg['student_dim'])
            self.encoder = nn.Sequential(nn.Linear(cfg['student_dim'], cfg['student_hidden']), nn.ReLU(),
                                         nn.Dropout(cfg['student_dropout']), nn.Linear(cfg['student_hidden'], cfg['student_dim']))
            nn.init.normal_(self.source_items.weight, std=.1)
            with torch.no_grad():
                self.source_items.weight[0].zero_()
            nn.init.normal_(self.target_items.weight, std=.1)
            # Zero correction with a trainable final layer; earlier gradients activate after one step.
            nn.init.zeros_(self.encoder[-1].weight)
            nn.init.zeros_(self.encoder[-1].bias)
            w = 1/torch.log2(torch.arange(cfg['k'], device=values.device)+2.)
            self.register_buffer('ranks', w/w.sum())

    def forward(self, x):
        u, i = x[:, 0], x[:, 1]
        correction = self.bias(i).squeeze(1)
        if self.arm != 'target_bias':
            known_user = u < len(self.lists)
            rows = self.lists[u.clamp(max=len(self.lists)-1)]
            h = self.source_items(rows, per_sample_weights=self.ranks.expand_as(rows))
            z = self.encoder(h)*known_user[:, None]
            correction = correction+(z*self.target_items(i)).sum(1)
        return self.baseline[i]+correction*self.seen[i]


@torch.no_grad()
def infer(model, x):
    model.eval()
    return torch.cat([model(b) for b in x.split(4096)]).cpu().numpy()


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def learn(task, cfg, manifest, seed, out):
    guard_test_labels(task, student=True)
    assert not (out/'selection.json').exists()
    assert digest(out/'topk.npz') == json.loads((out/'source_complete.json').read_text())['topk_sha256']
    train, val = load_frame(task, 'train'), load_frame(task, 'validation')
    assert not set(train.uid) & set(val.uid)
    api = np.load(out/'topk.npz')
    np.testing.assert_array_equal(api['user_ids'], np.arange(manifest['overlap_users']))
    vocab, compact = np.unique(api['item_ids'], return_inverse=True)
    compact = compact.reshape(api['item_ids'].shape)+1
    permutation = np.arange(len(compact))
    rng = np.random.default_rng(seed+991)
    for group in ['train', 'validation', 'test']:
        p = task/f'data/{group}_users.npy'
        assert digest(p) == manifest['hashes'][f'data/{group}_users.npy']
        users = np.load(p)
        order = rng.permutation(users)
        permutation[order] = np.roll(order, 1)
        assert np.all(permutation[users] != users)
        np.testing.assert_array_equal(np.sort(permutation[users]), np.sort(users))
    np.save(out/'permutation.npy', permutation)
    x = torch.tensor(train[['uid', 'iid']].to_numpy(), dtype=torch.long, device='cuda')
    y = torch.tensor(train.y.to_numpy(), dtype=torch.float32, device='cuda')
    vx = torch.tensor(val[['uid', 'iid']].to_numpy(), dtype=torch.long, device='cuda')
    for filename in ['values.npy', 'seen.npy']:
        assert digest(task/'baseline'/filename) == manifest['baseline_hashes'][f'baseline/{filename}']
    values = torch.tensor(np.load(task/'baseline/values.npy'), device='cuda')
    seen = torch.tensor(np.load(task/'baseline/seen.npy'), device='cuda')
    selection = {}
    init_hash = None
    for arm in cfg['arms']:
        seed_all(seed+200)
        rows = compact[permutation] if arm == 'shuffled' else compact
        lists = torch.tensor(rows, dtype=torch.long, device='cuda')
        model = Residual(values, seen, lists, len(vocab), cfg, arm).cuda()
        initial_hash = state_hash(model)
        if arm == 'real':
            init_hash = initial_hash
        if arm == 'shuffled':
            assert initial_hash == init_hash, 'Paired model initialization differs'
        initial = infer(model, vx)
        np.testing.assert_array_equal(initial, values[val.iid.to_numpy()].cpu().numpy())
        best = float(np.sqrt(np.square(initial-val.y.to_numpy()).mean()))
        chosen, history = 0, [dict(epoch=0, validation_rmse=best)]
        checkpoint = out/f'{arm}.pt'
        torch.save(model.state_dict(), checkpoint)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['student_lr'], weight_decay=cfg['student_weight_decay'])
        batch_rng = torch.Generator(device='cuda').manual_seed(seed+300)
        for epoch in range(1, cfg['student_epochs']+1):
            tick = time.monotonic()
            model.train()
            total = torch.zeros((), device='cuda')
            for idx in torch.randperm(len(x), device='cuda', generator=batch_rng).split(cfg['student_batch']):
                opt.zero_grad(set_to_none=True)
                loss = (model(x[idx])-y[idx]).square().mean()
                loss.backward()
                opt.step()
                total += loss.detach()*len(idx)
            pv = infer(model, vx)
            rmse = float(np.sqrt(np.square(pv-val.y.to_numpy()).mean()))
            assert np.isfinite(rmse) and torch.isfinite(total)
            if rmse < best:
                best, chosen = rmse, epoch
                torch.save(model.state_dict(), checkpoint)
            row = dict(epoch=epoch, train_mse=total.item()/len(x), validation_rmse=rmse,
                       validation_mae=float(np.abs(pv-val.y.to_numpy()).mean()), seconds=time.monotonic()-tick)
            history.append(row)
            save(out/f'{arm}_training.json', history)
            print('STUDENT_EPOCH', task.name, seed, arm, json.dumps(row), flush=True)
        model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
        pv = infer(model, vx)
        np.save(out/f'{arm}_validation_predictions.npy', pv)
        # Freeze before inference; test features have IDs only, labels remain inaccessible.
        info = dict(epoch=chosen, validation_rmse=best, initial_state_sha256=initial_hash,
                    checkpoint_sha256=digest(checkpoint), selection_metric='validation_rmse',
                    epochs_trained=cfg['student_epochs'], test_labels_read=False)
        save(out/f'{arm}_selection.json', info)
        fp = task/'data/test_features.npy'
        assert digest(fp) == manifest['hashes']['data/test_features.npy']
        tx = torch.tensor(np.load(fp), dtype=torch.long, device='cuda')
        prediction = infer(model, tx)
        assert prediction.shape == (len(tx),) and np.isfinite(prediction).all()
        np.testing.assert_array_equal(prediction[~seen[tx[:, 1]].cpu().numpy()],
                                      values[tx[:, 1]][~seen[tx[:, 1]]].cpu().numpy())
        np.save(out/f'{arm}_test_predictions.npy', prediction)
        info['predictions_sha256'] = digest(out/f'{arm}_test_predictions.npy')
        selection[arm] = info
        del model, opt, tx, lists
        torch.cuda.empty_cache()
    selection['checks'] = dict(zero_initial_correction=True, paired_initialization=True,
                              matched_training_shuffle=True, permutation_within_splits=True,
                              test_label_guard_enabled=True, source_weight_guard_enabled=True,
                              unseen_items_use_fixed_baseline=True)
    save(out/'selection.json', selection)
    print('ALL_STUDENTS_FROZEN', task.name, seed, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['source', 'learn'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    tasks = ['book_movie', 'movie_book', 'movie_music_clean']
    cfg = json.loads((args.root/'experiment.json').read_text())
    task = args.root/tasks[args.index//len(cfg['seeds'])]
    seed = cfg['seeds'][args.index % len(cfg['seeds'])]
    manifest = json.loads((task/'manifest.json').read_text())
    for name, expected in cfg['code_sha256'].items():
        assert digest(BASE/'scripts'/name) == expected, 'Implementation changed after protocol freeze'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    out = task/f'runs/seed_{seed}'
    out.mkdir(exist_ok=args.stage == 'learn')
    if args.stage == 'source':
        save(out/'environment.json', dict(job_id=os.environ.get('SLURM_JOB_ID'),
                                         gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    globals()[args.stage](task, cfg, manifest, seed, out)


if __name__ == '__main__':
    main()
