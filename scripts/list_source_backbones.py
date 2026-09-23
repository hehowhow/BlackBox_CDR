"""Change only the source Top-100 teacher; retain the original LIST student."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from book_movie_protocol import BASE, digest, save, predict
from residual_experiment import seed_all, guard_test_labels
from residual_summary import metrics, paired_interval

PARENT = BASE/'runs/pfcdr_backbones_8589'
NAMES = ['MF', 'GMF', 'DNN']


def read(path):
    return json.loads(path.read_text())


def check_code(root):
    for p, h in read(root/'protocol.json')['code_hashes'].items():
        assert digest(Path(p)) == h, p


def student_guard(root):
    guard_test_labels(root, student=True)
    def hook(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if PARENT in p.parents or p.name in {'scores.npy', 'pretrained.pt'}:
                raise RuntimeError(f'Forbidden teacher/score access by LIST: {p}')
    sys.addaudithook(hook)


class ListStudent(nn.Module):
    def __init__(self, lists, vocabulary_size, item_count, mean, cfg):
        super().__init__()
        self.register_buffer('lists', lists, persistent=False)
        w = 1/torch.log2(torch.arange(lists.shape[1], device=lists.device)+2.)
        self.register_buffer('rank_weights', w/w.sum())
        self.source_items = nn.EmbeddingBag(vocabulary_size+1, cfg['dim'], mode='sum', padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(cfg['dim'], cfg['hidden']), nn.ReLU(),
                                     nn.Dropout(cfg['dropout']), nn.Linear(cfg['hidden'], cfg['dim']))
        self.target_items = nn.Embedding(item_count, cfg['dim'])
        self.item_bias = nn.Embedding(item_count, 1)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        nn.init.normal_(self.source_items.weight, std=.1)
        with torch.no_grad():
            self.source_items.weight[0].zero_()
        nn.init.normal_(self.target_items.weight, std=.1)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        rows = self.lists[users]
        h = self.source_items(rows, per_sample_weights=self.rank_weights.expand_as(rows))
        z = self.encoder(h)*(rows[:, :1] != 0)
        return self.mean+self.item_bias(items).squeeze(-1)+(z*self.target_items(items)).sum(-1)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for d in ['queries', 'students']:
        (root/d).mkdir()
    for d in ['data', 'baseline']:
        (root/d).symlink_to((PARENT/d).resolve(), target_is_directory=True)
    manifest = read(PARENT/'manifest.json')
    save(root/'manifest.json', manifest)
    prior = read(PARENT/'protocol.json')
    for p, h in {**prior['code_sha256'], **prior['upstream_sha256']}.items():
        assert digest(Path(p)) == h, p
    checkpoints = {}
    for name in NAMES:
        p = PARENT/name/'pretrained.pt'
        checkpoints[name] = dict(path=str(p), sha256=digest(p))
        assert checkpoints[name]['sha256'] == read(PARENT/name/'complete.json')['pretrained_sha256']
        (root/'queries'/name).mkdir()
        (root/'students'/name).mkdir()
    paths = [Path(__file__), BASE/'scripts/slurm_list_source_backbones.sh',
             BASE/'scripts/run_list_encoder.py', BASE/'scripts/residual_experiment.py',
             BASE/'scripts/residual_summary.py', BASE/'scripts/book_movie_protocol.py']
    cfg = dict(task='Book->Movie', backbones=NAMES, source_checkpoints=checkpoints,
               seed=2020, k=100, export_batch=128, source_dim=10, source_epochs=10,
               source_lr=.01, seen_item_filter=False, score_clipping=False,
               learner_uses_scores=False, dim=32, hidden=64, dropout=.1,
               epochs=10, lr=.001, weight_decay=.00001, batch_size=2048,
               selection='minimum validation RMSE over epochs 1..10; all arms frozen before test labels',
               student='original LIST: rank pooling -> 32/64/32 ReLU MLP -> item dot + mean + item bias',
               vocabulary='union of returned opaque IDs only; shared ID initialization across teachers',
               unknown_target_items='frozen smoothed-mean fallback only for items absent from target training',
               source_scope='same pretrained source models as PFCDR 8589; no target weights or source weights given to student',
               scope='one task and one seed; historical test set, exploratory source-backbone comparison',
               bootstrap_repeats=2000, bootstrap_seed=992,
               code_hashes={str(p.resolve()):digest(p) for p in paths})
    save(root/'protocol.json', cfg)
    print('PREPARED', root, flush=True)


@torch.no_grad()
def export(root, name):
    from pfcdr_backbone_study import make_model, effective_users, table
    cfg, manifest = read(root/'protocol.json'), read(root/'manifest.json')
    guard_test_labels(root)
    check_code(root)
    parent_cfg = read(PARENT/'protocol.json')
    for p, h in {**parent_cfg['code_sha256'], **parent_cfg['upstream_sha256']}.items():
        assert digest(Path(p)) == h
    assert digest(root/'data/source.csv') == manifest['hashes']['data/source.csv']
    record = cfg['source_checkpoints'][name]
    assert digest(Path(record['path'])) == record['sha256']
    out = root/'queries'/name
    assert not (out/'complete.json').exists()
    start_time = time.monotonic()
    seed_all(cfg['seed'])
    m = make_model(name, manifest, cfg['source_dim'])
    m.load_state_dict(torch.load(record['path'], map_location='cuda', weights_only=True))
    m.eval()
    u = effective_users(m, name, 'src')[:manifest['overlap_users']]
    v = table(m, name, 'src').iid_embedding.weight
    ids, scores = [], []
    for start in range(0, len(u), cfg['export_batch']):
        score = u[start:start+cfg['export_batch']]@v.T
        values, indices = score.topk(cfg['k'], dim=1, sorted=True)
        ids.append(indices.cpu().numpy().astype(np.int32))
        scores.append(values.cpu().numpy())
        if start % 4096 == 0:
            print('EXPORT', name, min(start+len(score), len(u)), '/', len(u), flush=True)
    ids, scores = np.concatenate(ids), np.concatenate(scores)
    assert ids.shape == scores.shape == (manifest['overlap_users'], cfg['k'])
    assert np.isfinite(scores).all() and (np.diff(scores, axis=1) <= 0).all()
    assert (np.diff(np.sort(ids, axis=1), axis=1) > 0).all()
    # Independently score the complete catalog using the official forward for
    # four users. This catches missing GMF weights and missing DNN transforms.
    checks = []
    for uid in [0, len(u)//3, len(u)//2, len(u)-1]:
        native = []
        for first in range(0, len(v), 65536):
            items = torch.arange(first, min(first+65536, len(v)), device='cuda')
            x = torch.stack([torch.full_like(items, uid), items], 1)
            native.append(m(x, 'train_src'))
        native = torch.cat(native)
        fast = u[uid]@v.T
        torch.testing.assert_close(native, fast, rtol=3e-5, atol=3e-5)
        idx = torch.tensor(ids[uid], dtype=torch.long, device='cuda')
        reported = torch.tensor(scores[uid], device='cuda')
        torch.testing.assert_close(native[idx], reported, rtol=3e-5, atol=3e-5)
        kth = native.topk(cfg['k']).values[-1]
        assert native[idx].min() >= kth-3e-5*(1+kth.abs())
        native_ids = native.topk(cfg['k']).indices.cpu().numpy()
        checks.append(dict(user=uid, full_catalog=True, max_score_error=float((native-fast).abs().max()),
                           top100_native_overlap=int(np.isin(ids[uid], native_ids).sum())))
    np.savez_compressed(out/'topk_ids.npz', user_ids=np.arange(len(u), dtype=np.int32), item_ids=ids)
    np.save(out/'scores.npy', scores)
    save(out/'complete.json', dict(status='PASS', backbone=name, k=cfg['k'], users=len(u),
                                   source_checkpoint=record, item_ids_sha256=digest(out/'topk_ids.npz'),
                                   scores_sha256=digest(out/'scores.npy'), source_items=len(v),
                                   unique_returned_items=len(np.unique(ids)), seen_item_filter=False,
                                   score_clipping=False, native_checks=checks,
                                   seconds=time.monotonic()-start_time))
    print('EXPORT_PASS', name, flush=True)


def lists_for(root, name, vocabulary, cfg, manifest):
    out = root/'queries'/name
    assert digest(out/'topk_ids.npz') == read(out/'complete.json')['item_ids_sha256']
    with np.load(out/'topk_ids.npz') as api:
        np.testing.assert_array_equal(api['user_ids'], np.arange(manifest['overlap_users']))
        ids = api['item_ids']
    compact = np.searchsorted(vocabulary, ids)
    np.testing.assert_array_equal(vocabulary[compact], ids)
    lists = torch.zeros((manifest['target_users'], cfg['k']), dtype=torch.long, device='cuda')
    lists[:len(ids)] = torch.tensor(compact+1, device='cuda')
    return lists


def shared(root):
    student_guard(root)
    check_code(root)
    cfg, manifest = read(root/'protocol.json'), read(root/'manifest.json')
    assert not (root/'shared.json').exists()
    ids = {}
    for name in NAMES:
        p = root/'queries'/name/'topk_ids.npz'
        assert digest(p) == read(p.parent/'complete.json')['item_ids_sha256']
        ids[name] = np.load(p)['item_ids']
    vocab = np.unique(np.concatenate([x.ravel() for x in ids.values()]))
    np.save(root/'vocabulary.npy', vocab)
    comparison = {}
    for a, b in itertools.combinations(NAMES, 2):
        merged = np.sort(np.concatenate([ids[a], ids[b]], axis=1), axis=1)
        common = (np.diff(merged, axis=1) == 0).sum(1)
        comparison[a+'__'+b] = dict(mean_common_items=float(common.mean()),
                                    mean_overlap_fraction=float(common.mean()/cfg['k']),
                                    mean_jaccard=float((common/(2*cfg['k']-common)).mean()),
                                    same_top1_fraction=float((ids[a][:, 0] == ids[b][:, 0]).mean()))
    assert digest(root/'data/train.csv') == manifest['hashes']['data/train.csv']
    train = pd.read_csv(root/'data/train.csv', header=None).to_numpy()
    mean = torch.tensor(train[:, 2], dtype=torch.float32).mean().item()
    seed_all(cfg['seed'])
    model = ListStudent(lists_for(root, 'MF', vocab, cfg, manifest), len(vocab), manifest['target_items'], mean, cfg).cuda()
    torch.save(model.state_dict(), root/'initial_student.pt')
    save(root/'shared.json', dict(initial_sha256=digest(root/'initial_student.pt'),
                                  vocabulary_sha256=digest(root/'vocabulary.npy'),
                                  vocabulary_size=len(vocab), train_mean=mean, list_comparisons=comparison))
    print('SHARED_INIT', json.dumps(comparison), flush=True)


@torch.no_grad()
def infer(model, x, seen, fallback):
    model.eval()
    raw = torch.cat([model(b[:, 0], b[:, 1]) for b in x.split(4096)])
    return torch.where(seen[x[:, 1]], raw, fallback[x[:, 1]]).cpu().numpy()


def load_student(root, name, cfg, manifest):
    shared_info = read(root/'shared.json')
    assert digest(root/'vocabulary.npy') == shared_info['vocabulary_sha256']
    vocab = np.load(root/'vocabulary.npy')
    model = ListStudent(lists_for(root, name, vocab, cfg, manifest), len(vocab), manifest['target_items'], shared_info['train_mean'], cfg).cuda()
    return model


def train(root, name):
    student_guard(root)
    check_code(root)
    cfg, manifest = read(root/'protocol.json'), read(root/'manifest.json')
    out = root/'students'/name
    assert not (out/'started.json').exists()
    save(out/'started.json', dict(job=os.environ['SLURM_JOB_ID'], gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                                 device=torch.cuda.get_device_name(), torch=torch.__version__))
    for p in ['data/train.csv', 'data/validation.csv', 'data/train_users.npy', 'data/validation_users.npy',
              'data/test_users.npy', 'data/test_features.npy']:
        assert digest(root/p) == manifest['hashes'][p]
    for p, h in manifest['baseline_hashes'].items():
        assert digest(root/p) == h
    tr = pd.read_csv(root/'data/train.csv', header=None).to_numpy()
    va = pd.read_csv(root/'data/validation.csv', header=None).to_numpy()
    vu, tu = np.load(root/'data/validation_users.npy'), np.load(root/'data/test_users.npy')
    assert not np.intersect1d(tr[:, 0], np.concatenate([vu, tu])).size
    np.testing.assert_array_equal(np.unique(va[:, 0]), vu)
    seed_all(cfg['seed'])
    model = load_student(root, name, cfg, manifest)
    init_hash = digest(root/'initial_student.pt')
    assert init_hash == read(root/'shared.json')['initial_sha256']
    model.load_state_dict(torch.load(root/'initial_student.pt', map_location='cuda', weights_only=True))
    # Reset dropout RNG after construction; all arms use the same shape and batches.
    seed_all(cfg['seed']+100)
    x = torch.tensor(tr[:, :2], dtype=torch.long, device='cuda')
    y = torch.tensor(tr[:, 2], dtype=torch.float32, device='cuda')
    vx = torch.tensor(va[:, :2], dtype=torch.long, device='cuda')
    seen = torch.tensor(np.load(root/'baseline/seen.npy'), device='cuda')
    fallback = torch.tensor(np.load(root/'baseline/values.npy'), device='cuda')
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator(device='cuda').manual_seed(cfg['seed']+200)
    order_hash = hashlib.sha256()
    history, selected = [], None
    for epoch in range(1, cfg['epochs']+1):
        tick = time.monotonic()
        model.train()
        total = torch.zeros((), device='cuda')
        order = torch.randperm(len(x), generator=generator, device='cuda')
        order_hash.update(order.cpu().numpy().tobytes())
        for idx in order.split(cfg['batch_size']):
            opt.zero_grad(set_to_none=True)
            loss = (model(x[idx, 0], x[idx, 1])-y[idx]).square().mean()
            loss.backward()
            opt.step()
            total += loss.detach()*len(idx)
        pred = infer(model, vx, seen, fallback)
        rmse = float(np.sqrt(np.square(pred-va[:, 2]).mean()))
        mae = float(np.abs(pred-va[:, 2]).mean())
        row = dict(epoch=epoch, train_mse=total.item()/len(x), validation_rmse=rmse, validation_mae=mae,
                   seconds=time.monotonic()-tick)
        assert all(np.isfinite(row[k]) for k in ['train_mse', 'validation_rmse', 'validation_mae'])
        if selected is None or rmse < selected['validation_rmse']:
            selected = dict(row)
            torch.save(model.state_dict(), out/'selected.pt')
            np.save(out/'validation.npy', pred)
        history.append(row)
        save(out/'history.json', history)
        print('LIST_EPOCH', name, json.dumps(row), flush=True)
    model.load_state_dict(torch.load(out/'selected.pt', map_location='cuda', weights_only=True))
    np.testing.assert_array_equal(infer(model, vx, seen, fallback), np.load(out/'validation.npy'))
    tx = torch.tensor(np.load(root/'data/test_features.npy'), dtype=torch.long, device='cuda')
    pred = infer(model, tx, seen, fallback)
    assert np.isfinite(pred).all()
    np.save(out/'test.npy', pred)
    selected.update(initial_sha256=init_hash, batch_order_sha256=order_hash.hexdigest(),
                    checkpoint_sha256=digest(out/'selected.pt'), validation_sha256=digest(out/'validation.npy'),
                    test_sha256=digest(out/'test.npy'), list_sha256=digest(root/'queries'/name/'topk_ids.npz'))
    save(out/'selection.json', selected)
    save(out/'complete.json', dict(status='PASS', selection_sha256=digest(out/'selection.json')))
    print('LIST_SELECTION_FROZEN', name, json.dumps(selected), flush=True)


def selftest(root):
    # Compare our explicit shared-vocabulary implementation with the original
    # class, including loss gradients and the no-source-list behavior.
    from run_list_encoder import ListRecommender, CFG
    cfg = read(root/'protocol.json')
    for key in ['dim', 'hidden', 'dropout']:
        assert cfg[key] == CFG[key]
    seed_all(cfg['seed'])
    lists = torch.tensor([[1, 2, 3], [3, 2, 1], [0, 0, 0]], device='cuda')
    a = ListRecommender(lists, 5, 4.).cuda()
    b = ListStudent(lists, 3, 5, 4., cfg).cuda()
    b.load_state_dict(a.state_dict())
    u, i = torch.arange(3, device='cuda'), torch.arange(3, device='cuda')
    a.eval(); b.eval()
    pa, pb = a(u, i), b(u, i)
    torch.testing.assert_close(pa, pb, rtol=0, atol=0)
    (pa-3).square().mean().backward(); (pb-3).square().mean().backward()
    for (na, va), (nb, vb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        torch.testing.assert_close(va.grad, vb.grad, rtol=0, atol=0)
    assert pb[-1] == b.mean+b.item_bias(i[-1]).squeeze()
    save(root/'selftest.json', dict(status='PASS', original_list_forward=True,
                                    original_list_gradients=True, no_list_user_bias_only=True))
    print('LIST_SELFTEST_PASS', flush=True)


def verify(root):
    student_guard(root)
    check_code(root)
    cfg, manifest = read(root/'protocol.json'), read(root/'manifest.json')
    assert read(root/'selftest.json')['status'] == 'PASS'
    seen = torch.tensor(np.load(root/'baseline/seen.npy'), device='cuda')
    fallback = torch.tensor(np.load(root/'baseline/values.npy'), device='cuda')
    va = pd.read_csv(root/'data/validation.csv', header=None).to_numpy()
    vx = torch.tensor(va[:, :2], dtype=torch.long, device='cuda')
    tx = torch.tensor(np.load(root/'data/test_features.npy'), dtype=torch.long, device='cuda')
    initial, orders, checks = set(), set(), []
    for name in NAMES:
        out = root/'students'/name
        assert read(out/'complete.json')['selection_sha256'] == digest(out/'selection.json')
        s = read(out/'selection.json')
        for filename, key in [('selected.pt', 'checkpoint_sha256'), ('validation.npy', 'validation_sha256'), ('test.npy', 'test_sha256')]:
            assert digest(out/filename) == s[key]
        assert s['list_sha256'] == digest(root/'queries'/name/'topk_ids.npz')
        h = read(out/'history.json')
        assert [v['epoch'] for v in h] == list(range(1, cfg['epochs']+1))
        assert s['epoch'] == min(h, key=lambda v:v['validation_rmse'])['epoch']
        initial.add(s['initial_sha256']); orders.add(s['batch_order_sha256'])
        model = load_student(root, name, cfg, manifest)
        model.load_state_dict(torch.load(out/'selected.pt', map_location='cuda', weights_only=True))
        vp = infer(model, vx, seen, fallback)
        np.testing.assert_array_equal(vp, np.load(out/'validation.npy'))
        np.testing.assert_array_equal(infer(model, tx, seen, fallback), np.load(out/'test.npy'))
        assert abs(float(np.sqrt(np.square(vp-va[:, 2]).mean()))-s['validation_rmse']) < 1e-12
        checks.append(dict(backbone=name, selected_epoch=s['epoch'], restored_predictions=True))
        del model
    assert len(initial) == len(orders) == 1
    save(root/'verification.json', dict(status='PASS', shared_initialization=True, shared_batch_order=True, checks=checks))
    print('LIST_VERIFY_PASS', flush=True)


def evaluate(root):
    check_code(root)
    cfg, manifest = read(root/'protocol.json'), read(root/'manifest.json')
    assert read(root/'verification.json')['status'] == 'PASS'
    frozen = {n:digest(root/'students'/n/'selection.json') for n in NAMES}
    with (root/'evaluation_started.json').open('x') as f:
        json.dump(frozen, f, indent=2)
    assert digest(root/'data/test.csv') == manifest['hashes']['data/test.csv']
    test = pd.read_csv(root/'data/test.csv', header=None).to_numpy()
    np.testing.assert_array_equal(test[:, :2], np.load(root/'data/test_features.npy'))
    y = test[:, 2]
    _, users = np.unique(test[:, 0], return_inverse=True)
    stats = dict(np.load(root/'baseline/statistics.npz'))
    baselines = {n:predict(stats, test[:, 1].astype(int), n) for n in ['item_mean', 'smoothed_mean_5', 'item_median']}
    models, rows = {}, []
    mf = np.load(root/'students/MF/test.npy')
    for name in NAMES:
        out = root/'students'/name
        s = read(out/'selection.json')
        assert digest(out/'test.npy') == s['test_sha256']
        p = np.load(out/'test.npy')
        m = metrics(p, y, users)
        comparisons = {n:paired_interval(p[None, :], ref[None, :], y, users, cfg)
                       for n, ref in {**baselines, 'LIST_MF':mf}.items()}
        for v in comparisons.values():
            v['interpretation'] = 'negative favors this LIST arm; user-cluster CI conditional on one fitted seed'
        models[name] = dict(selection=s, metrics=m, comparisons=comparisons)
        rows.append(dict(source_backbone=name, k=cfg['k'], epoch=s['epoch'], mae=m['mae'], rmse=m['rmse']))
    result = dict(status='PASS', protocol=cfg, shared=read(root/'shared.json'),
                  baselines={n:metrics(p, y, users) for n, p in baselines.items()}, models=models)
    save(root/'summary.json', result)
    import csv
    with (root/'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ['# LIST 源域 backbone 比较', '',
             'Book→Movie；Top-100；单种子 2020；目标 LIST 结构和初始参数固定；验证 RMSE 选 epoch。', '',
             '| 方法 | epoch | MAE | RMSE |', '| --- | ---: | ---: | ---: |']
    for n, v in result['baselines'].items():
        lines.append(f"| {n} | — | {v['mae']:.6f} | {v['rmse']:.6f} |")
    for r in rows:
        lines.append(f"| LIST，源 {r['source_backbone']} | {r['epoch']} | {r['mae']:.6f} | {r['rmse']:.6f} |")
    lines += ['', '此轮没有改变 LIST 的目标评分结构，也没有使用 PFCDR 的目标模型、反演原型或投票结果。',
              '源域 checkpoint 复用 8589；学生只读各自的 Top-100 ID 列表，分数与源模型权重对其不可读。',
              '共用三种 API 返回 ID 的并集词表；该词表不含额外源交互或评分。目标无源列表用户仅贡献物品偏置预测。',
              '保留未过滤已交互源物品的既有 API 规则，不裁剪评分；训练未见目标物品回退固定平滑均值。',
              '所有三个模型选定并恢复核验后才统一读取测试标签。历史测试集已用过，本轮是探索性结果。',
              '所有配对区间、列表重叠、代码和 checkpoint 哈希见 summary.json / protocol.json / queries/*/complete.json。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('LIST_SOURCE_STUDY_PASS', json.dumps(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'export', 'shared', 'selftest', 'train', 'verify', 'evaluate'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--backbone', choices=NAMES)
    args = p.parse_args()
    root = args.root.resolve()
    if args.stage in ['export', 'train']:
        assert args.backbone is not None
        globals()[args.stage](root, args.backbone)
    else:
        globals()[args.stage](root)
