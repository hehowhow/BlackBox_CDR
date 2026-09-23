"""Fixed MF API lists, matched LIST encoders, five target scoring heads."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from book_movie_protocol import BASE, digest, save, predict
from residual_experiment import seed_all
from residual_summary import metrics, paired_interval
from list_source_backbones import ListStudent, student_guard, infer, read, check_code, lists_for

PRIOR = BASE/'runs/list_source_backbones_8596'
HEADS = ['DOT', 'GMF', 'DNN', 'MLP', 'DOT_MLP']
SEEDS = [2020, 2021, 2022]


class ScoredList(ListStudent):
    def __init__(self, lists, vocab, items, mean, cfg, head):
        super().__init__(lists, vocab, items, mean, cfg)
        self.kind = head
        d = cfg['dim']
        # Head initialization uses its own RNG; shared parameters are restored
        # from the identical per-seed checkpoint before learning.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['seed']+3000)
            if head == 'GMF':
                self.head = nn.Linear(d, 1, bias=False)
                nn.init.ones_(self.head.weight)
            elif head == 'DNN':
                self.head = nn.Linear(d, d)
                nn.init.eye_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            elif head in ['MLP', 'DOT_MLP']:
                self.head = nn.Sequential(nn.Linear(2*d, 64), nn.ReLU(),
                                          nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
                # Small random MLP output; hybrid starts at the original dot
                # function via a zero final residual layer.
                if head == 'DOT_MLP':
                    nn.init.zeros_(self.head[-1].weight)
                    nn.init.zeros_(self.head[-1].bias)

    def interaction(self, z, v):
        if self.kind == 'DOT': return (z*v).sum(-1)
        if self.kind == 'GMF': return self.head(z*v).squeeze(-1)
        if self.kind == 'DNN': return (self.head(z)*v).sum(-1)
        nonlinear = self.head(torch.cat([z, v], -1)).squeeze(-1)
        return nonlinear if self.kind == 'MLP' else (z*v).sum(-1)+nonlinear

    def forward(self, users, items):
        rows = self.lists[users]
        h = self.source_items(rows, per_sample_weights=self.rank_weights.expand_as(rows))
        mask = rows[:, :1] != 0
        z = self.encoder(h)*mask
        # Apply the list mask AFTER scoring too: affine and nonlinear biases
        # must not introduce new target-only-user behavior across heads.
        score = self.interaction(z, self.target_items(items))*mask.squeeze(-1)
        return self.mean+self.item_bias(items).squeeze(-1)+score


def construct(root, name, cfg, manifest):
    shared = read(root/'shared.json')
    assert digest(root/'vocabulary.npy') == shared['vocabulary_sha256']
    vocab = np.load(root/'vocabulary.npy')
    lists = lists_for(root, 'MF', vocab, cfg, manifest)
    return ScoredList(lists, len(vocab), manifest['target_items'], shared['train_mean'], cfg, name).cuda()


def load_initial(model, root):
    state = torch.load(root/'initial_student.pt', map_location='cuda', weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert set(missing) == {k for k in model.state_dict() if k.startswith('head.')}
    for k, v in state.items():
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    check_code(PRIOR)
    prior_cfg = read(PRIOR/'protocol.json')
    prior_shared = read(PRIOR/'shared.json')
    manifest = read(PRIOR/'manifest.json')
    assert digest(PRIOR/'initial_student.pt') == prior_shared['initial_sha256']
    assert digest(PRIOR/'vocabulary.npy') == prior_shared['vocabulary_sha256']
    mf = PRIOR/'queries/MF'
    assert digest(mf/'topk_ids.npz') == read(mf/'complete.json')['item_ids_sha256']
    paths = [Path(__file__), BASE/'scripts/slurm_list_target_heads.sh',
             BASE/'scripts/list_source_backbones.py', BASE/'scripts/book_movie_protocol.py',
             BASE/'scripts/residual_experiment.py', BASE/'scripts/residual_summary.py']
    cfg = {k:prior_cfg[k] for k in ['dim', 'hidden', 'dropout', 'epochs', 'lr', 'weight_decay', 'batch_size', 'k']}
    cfg.update(task='Book->Movie', source_backbone='MF', source_seed=2020, heads=HEADS, seeds=SEEDS,
               source_lists_sha256=digest(mf/'topk_ids.npz'), source_list_origin=str(mf/'topk_ids.npz'),
               source_list_policy='unchanged fixed Top100; no scores; no seen-item filtering',
               encoder='same trainable rank-pooling MLP; architecture and initial parameters matched within seed',
               head_definitions=dict(DOT='sum(z*v)', GMF='w^T(z*v), no bias',
                                     DNN='(Az+b)^T v, author-style affine user transform',
                                     MLP='concat(z,v): 64->64 ReLU->32 ReLU->1',
                                     DOT_MLP='sum(z*v) + same nonlinear MLP residual; not official NeuMF'),
               head_initialization='GMF ones; DNN identity/zero bias; DOT_MLP zero last layer; MLP default random',
               common_prediction='training global mean + trainable item bias + masked interaction',
               no_list_users='interaction forced zero after head; mean + item bias only',
               vocabulary='reuse 8596 shared opaque-ID vocabulary for exact seed-2020 control; only MF lists are read',
               selection='minimum validation RMSE among epochs 1..10, independently per head/seed',
               selection_of_method='mean selected validation RMSE across three seeds, frozen before test labels',
               unknown_target_items='same frozen smoothed-mean fallback for items absent from target training',
               test_policy='all 15 runs frozen and independently verified before test labels',
               scope='one task, one fixed source model/list, three student seeds; no learning-rate or epoch-budget search',
               bootstrap_repeats=2000, bootstrap_seed=993,
               code_hashes={str(p.resolve()):digest(p) for p in paths})
    save(root/'protocol.json', cfg)
    save(root/'manifest.json', manifest)
    for seed in SEEDS:
        sub = root/f'seed_{seed}'
        sub.mkdir()
        for d in ['data', 'baseline']:
            (sub/d).symlink_to((PRIOR/d).resolve(), target_is_directory=True)
        (sub/'queries').mkdir()
        (sub/'queries/MF').symlink_to(mf, target_is_directory=True)
        (sub/'vocabulary.npy').symlink_to(PRIOR/'vocabulary.npy')
        (sub/'students').mkdir()
        for name in HEADS: (sub/'students'/name).mkdir()
        save(sub/'manifest.json', manifest)
        scfg = dict(cfg, seed=seed)
        save(sub/'protocol.json', scfg)
        save(sub/'shared.json', prior_shared)
        if seed == 2020:
            shutil.copyfile(PRIOR/'initial_student.pt', sub/'initial_student.pt')
        else:
            seed_all(seed)
            vocab = np.load(sub/'vocabulary.npy')
            lists = lists_for(sub, 'MF', vocab, scfg, manifest)
            model = ListStudent(lists, len(vocab), manifest['target_items'], prior_shared['train_mean'], scfg).cuda()
            torch.save(model.state_dict(), sub/'initial_student.pt')
            del model, lists
        save(sub/'shared.json', dict(prior_shared, initial_sha256=digest(sub/'initial_student.pt')))
    print('PREPARED', root, flush=True)


def selftest(root):
    cfg = dict(read(root/'protocol.json'), seed=2020)
    seed_all(731)
    lists = torch.tensor([[1, 2, 3], [3, 1, 2], [0, 0, 0]], device='cuda')
    old = ListStudent(lists, 3, 5, 4., cfg).cuda().eval()
    users = torch.arange(3, device='cuda')
    items = torch.arange(3, device='cuda')
    checks = []
    for name in HEADS:
        m = ScoredList(lists, 3, 5, 4., cfg, name).cuda().eval()
        m.load_state_dict(old.state_dict(), strict=False)
        p = m(users, items)
        if name != 'MLP':
            torch.testing.assert_close(p, old(users, items), rtol=1e-6, atol=1e-6)
        # Validate formulas and input gradients with nontrivial head weights.
        if name in ['GMF', 'DNN']:
            with torch.no_grad():
                for w in m.head.parameters(): w.normal_(0, .3)
        z = torch.randn(4, cfg['dim'], device='cuda', requires_grad=True)
        v = torch.randn_like(z, requires_grad=True)
        actual = m.interaction(z, v)
        if name == 'DOT': expected = torch.einsum('bi,bi->b', z, v)
        elif name == 'GMF': expected = (z*v*m.head.weight[0]).sum(-1)
        elif name == 'DNN': expected = ((z@m.head.weight.T+m.head.bias)*v).sum(-1)
        else:
            layers = m.head
            h = torch.relu(torch.cat([z, v], -1)@layers[0].weight.T+layers[0].bias)
            h = torch.relu(h@layers[2].weight.T+layers[2].bias)
            expected = (h@layers[4].weight.T+layers[4].bias).squeeze(-1)
            if name == 'DOT_MLP': expected = expected+(z*v).sum(-1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        ga = torch.autograd.grad(actual.square().sum(), (z, v), retain_graph=True)
        ge = torch.autograd.grad(expected.square().sum(), (z, v))
        for a, b in zip(ga, ge): torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        # No-list users must not update the head, encoder, or target vectors.
        m.zero_grad(set_to_none=True)
        p = m(users[-1:], items[-1:])
        torch.testing.assert_close(p, m.mean+m.item_bias(items[-1:]).squeeze(-1), rtol=0, atol=0)
        (p-2).square().sum().backward()
        for key, parameter in m.named_parameters():
            if key != 'item_bias.weight' and parameter.grad is not None:
                assert not torch.count_nonzero(parameter.grad), key
        checks.append(dict(head=name, formula=True, input_gradients=True, no_list_bias_only=True,
                           dot_function_initialization=name != 'MLP'))
    save(root/'selftest.json', dict(status='PASS', checks=checks))
    print('HEAD_SELFTEST_PASS', flush=True)


def train(root, seed, name):
    sub = root/f'seed_{seed}'
    student_guard(sub)
    check_code(sub)
    cfg, manifest = read(sub/'protocol.json'), read(sub/'manifest.json')
    out = sub/'students'/name
    assert not (out/'started.json').exists()
    save(out/'started.json', dict(job=os.environ['SLURM_JOB_ID'], gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                                 device=torch.cuda.get_device_name(), torch=torch.__version__, seed=seed, head=name))
    for p in ['data/train.csv', 'data/validation.csv', 'data/train_users.npy', 'data/validation_users.npy',
              'data/test_users.npy', 'data/test_features.npy']:
        assert digest(sub/p) == manifest['hashes'][p]
    for p, h in manifest['baseline_hashes'].items(): assert digest(sub/p) == h
    assert digest(sub/'queries/MF/topk_ids.npz') == cfg['source_lists_sha256']
    tr = pd.read_csv(sub/'data/train.csv', header=None).to_numpy()
    va = pd.read_csv(sub/'data/validation.csv', header=None).to_numpy()
    vu, tu = np.load(sub/'data/validation_users.npy'), np.load(sub/'data/test_users.npy')
    assert not np.intersect1d(tr[:, 0], np.concatenate([vu, tu])).size
    np.testing.assert_array_equal(np.unique(va[:, 0]), vu)
    np.testing.assert_array_equal(np.unique(tr[tr[:, 0] < manifest['overlap_users'], 0]), np.load(sub/'data/train_users.npy'))
    seed_all(seed)
    m = construct(sub, name, cfg, manifest)
    init_hash = digest(sub/'initial_student.pt')
    assert init_hash == read(sub/'shared.json')['initial_sha256']
    load_initial(m, sub)
    parameter_count = sum(v.numel() for v in m.parameters())
    # Only the encoder has dropout. Extra heads have no random layers.
    seed_all(seed+100)
    x = torch.tensor(tr[:, :2], dtype=torch.long, device='cuda')
    y = torch.tensor(tr[:, 2], dtype=torch.float32, device='cuda')
    vx = torch.tensor(va[:, :2], dtype=torch.long, device='cuda')
    seen = torch.tensor(np.load(sub/'baseline/seen.npy'), device='cuda')
    fallback = torch.tensor(np.load(sub/'baseline/values.npy'), device='cuda')
    opt = torch.optim.Adam(m.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator(device='cuda').manual_seed(seed+200)
    order_hash = hashlib.sha256()
    history, selected = [], None
    for epoch in range(1, cfg['epochs']+1):
        tick = time.monotonic()
        m.train()
        total = torch.zeros((), device='cuda')
        order = torch.randperm(len(x), generator=generator, device='cuda')
        order_hash.update(order.cpu().numpy().tobytes())
        for idx in order.split(cfg['batch_size']):
            opt.zero_grad(set_to_none=True)
            loss = (m(x[idx, 0], x[idx, 1])-y[idx]).square().mean()
            loss.backward()
            opt.step()
            total += loss.detach()*len(idx)
        p = infer(m, vx, seen, fallback)
        rmse = float(np.sqrt(np.square(p-va[:, 2]).mean()))
        row = dict(epoch=epoch, train_mse=total.item()/len(x), validation_rmse=rmse,
                   validation_mae=float(np.abs(p-va[:, 2]).mean()), seconds=time.monotonic()-tick)
        assert all(np.isfinite(row[k]) for k in ['train_mse', 'validation_rmse', 'validation_mae'])
        if selected is None or rmse < selected['validation_rmse']:
            selected = dict(row)
            torch.save(m.state_dict(), out/'selected.pt')
            np.save(out/'validation.npy', p)
        history.append(row)
        save(out/'history.json', history)
        print('HEAD_EPOCH', seed, name, json.dumps(row), flush=True)
    m.load_state_dict(torch.load(out/'selected.pt', weights_only=True, map_location='cuda'))
    np.testing.assert_array_equal(infer(m, vx, seen, fallback), np.load(out/'validation.npy'))
    tx = torch.tensor(np.load(sub/'data/test_features.npy'), device='cuda', dtype=torch.long)
    p = infer(m, tx, seen, fallback)
    assert np.isfinite(p).all()
    np.save(out/'test.npy', p)
    selected.update(initial_sha256=init_hash, batch_order_sha256=order_hash.hexdigest(), parameters=parameter_count,
                    head_parameters=sum(v.numel() for k, v in m.named_parameters() if k.startswith('head.')),
                    checkpoint_sha256=digest(out/'selected.pt'), validation_sha256=digest(out/'validation.npy'),
                    test_sha256=digest(out/'test.npy'), list_sha256=cfg['source_lists_sha256'])
    save(out/'selection.json', selected)
    save(out/'complete.json', dict(status='PASS', selection_sha256=digest(out/'selection.json')))
    print('HEAD_SELECTION_FROZEN', seed, name, json.dumps(selected), flush=True)


def verify(root):
    student_guard(root)
    check_code(root)
    seed_all(2020)
    cfg, manifest = read(root/'protocol.json'), read(root/'manifest.json')
    assert read(root/'selftest.json')['status'] == 'PASS'
    reference = root/'seed_2020'
    va = pd.read_csv(reference/'data/validation.csv', header=None).to_numpy()
    vx = torch.tensor(va[:, :2], dtype=torch.long, device='cuda')
    tx = torch.tensor(np.load(reference/'data/test_features.npy'), dtype=torch.long, device='cuda')
    seen = torch.tensor(np.load(reference/'baseline/seen.npy'), device='cuda')
    fallback = torch.tensor(np.load(reference/'baseline/values.npy'), device='cuda')
    checks, validation = [], {n:[] for n in HEADS}
    for seed in SEEDS:
        sub = root/f'seed_{seed}'
        scfg = read(sub/'protocol.json')
        check_code(sub)
        assert {k:v for k, v in scfg.items() if k != 'seed'} == cfg
        initial, orders = set(), set()
        for name in HEADS:
            out = sub/'students'/name
            assert digest(out/'selection.json') == read(out/'complete.json')['selection_sha256']
            s = read(out/'selection.json')
            for p, key in [('selected.pt', 'checkpoint_sha256'), ('validation.npy', 'validation_sha256'), ('test.npy', 'test_sha256')]:
                assert digest(out/p) == s[key]
            assert s['list_sha256'] == cfg['source_lists_sha256'] == digest(sub/'queries/MF/topk_ids.npz')
            initial.add(s['initial_sha256']); orders.add(s['batch_order_sha256'])
            h = read(out/'history.json')
            assert [v['epoch'] for v in h] == list(range(1, cfg['epochs']+1))
            assert s['epoch'] == min(h, key=lambda v:v['validation_rmse'])['epoch']
            m = construct(sub, name, scfg, manifest)
            m.load_state_dict(torch.load(out/'selected.pt', weights_only=True, map_location='cuda'))
            vp = infer(m, vx, seen, fallback)
            np.testing.assert_array_equal(vp, np.load(out/'validation.npy'))
            np.testing.assert_array_equal(infer(m, tx, seen, fallback), np.load(out/'test.npy'))
            rmse = float(np.sqrt(np.square(vp-va[:, 2]).mean()))
            assert abs(rmse-s['validation_rmse']) < 1e-12
            validation[name].append(rmse)
            checks.append(dict(seed=seed, head=name, epoch=s['epoch'], full_prediction_restore=True))
            del m
        assert len(initial) == len(orders) == 1
        assert next(iter(initial)) == digest(sub/'initial_student.pt')
    # Regression comparison to the already completed MF-list, DOT-score run.
    replay = {}
    for name in ['validation.npy', 'test.npy']:
        previous = np.load(PRIOR/'students/MF'/name)
        current = np.load(reference/'students/DOT'/name)
        np.testing.assert_allclose(previous, current, rtol=1e-6, atol=1e-6)
        replay[name] = float(np.abs(previous-current).max())
    means = {n:float(np.mean(v)) for n, v in validation.items()}
    selected = min(HEADS, key=lambda n:means[n])
    save(root/'method_selection.json', dict(selected_head=selected, metric='mean selected validation RMSE',
                                           validation_means=means, test_labels_used=False))
    save(root/'verification.json', dict(status='PASS', checks=checks, matched_initialization_and_batches=True,
                                        original_MF_LIST_replay_max_error=replay,
                                        method_selection_sha256=digest(root/'method_selection.json')))
    print('HEAD_VERIFY_PASS', json.dumps(means), 'selected', selected, flush=True)


def paired_seed_mean(a, b, y, users, cfg):
    """Bootstrap user clusters, then average individual-seed MAE and RMSE.

    This matches the reported arithmetic mean of seed metrics, not an ensemble
    or the square root of an average of seed MSEs.
    """
    b = np.broadcast_to(b, a.shape)
    counts = np.bincount(users)
    errors = [np.abs(a-y), np.abs(b-y), np.square(a-y), np.square(b-y)]
    sums = np.array([[np.bincount(users, weights=row) for row in e] for e in errors])
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    draws = []
    for _ in range(cfg['bootstrap_repeats']):
        idx = rng.integers(len(counts), size=len(counts))
        aa, ab, sa, sb = sums[:, :, idx].sum(-1)/counts[idx].sum()
        draws.append([float((aa-ab).mean()), float((np.sqrt(sa)-np.sqrt(sb)).mean())])
    ci = np.quantile(draws, [.025, .975], axis=0)
    return dict(delta_mae=float(np.abs(a-y).mean()-np.abs(b-y).mean()),
                delta_rmse=float((np.sqrt(np.square(a-y).mean(1))-np.sqrt(np.square(b-y).mean(1))).mean()),
                mae_95ci=ci[:, 0].tolist(), rmse_95ci=ci[:, 1].tolist(),
                interpretation='negative favors head; average of individual-seed metrics; user-cluster CI conditional on three fitted students and one fixed source list')


def evaluate(root):
    check_code(root)
    cfg, manifest = read(root/'protocol.json'), read(root/'manifest.json')
    verification = read(root/'verification.json')
    assert verification['status'] == 'PASS'
    assert digest(root/'method_selection.json') == verification['method_selection_sha256']
    frozen = {}
    for seed in SEEDS:
        for name in HEADS:
            out = root/f'seed_{seed}/students'/name
            assert digest(out/'selection.json') == read(out/'complete.json')['selection_sha256']
            frozen[f'{seed}/{name}'] = digest(out/'selection.json')
    with (root/'evaluation_started.json').open('x') as f: json.dump(frozen, f, indent=2)
    reference = root/'seed_2020'
    assert digest(reference/'data/test.csv') == manifest['hashes']['data/test.csv']
    test = pd.read_csv(reference/'data/test.csv', header=None).to_numpy()
    np.testing.assert_array_equal(test[:, :2], np.load(reference/'data/test_features.npy'))
    y = test[:, 2]
    _, users = np.unique(test[:, 0], return_inverse=True)
    stats = dict(np.load(reference/'baseline/statistics.npz'))
    baselines = {n:predict(stats, test[:, 1].astype(int), n) for n in ['item_mean', 'smoothed_mean_5', 'item_median']}
    predictions, rows, models = {}, [], {}
    for name in HEADS:
        predictions[name] = np.stack([np.load(root/f'seed_{s}/students'/name/'test.npy') for s in SEEDS])
        per_seed = {}
        for k, seed in enumerate(SEEDS):
            out = root/f'seed_{seed}/students'/name
            s = read(out/'selection.json')
            assert digest(out/'test.npy') == s['test_sha256']
            m = metrics(predictions[name][k], y, users)
            per_seed[str(seed)] = dict(metrics=m, selection=s)
            rows.append(dict(head=name, seed=seed, epoch=s['epoch'], validation_rmse=s['validation_rmse'],
                             mae=m['mae'], rmse=m['rmse'], parameters=s['parameters'], head_parameters=s['head_parameters']))
        agg = {key:dict(mean=float(np.mean([p['metrics'][key] for p in per_seed.values()])),
                       std=float(np.std([p['metrics'][key] for p in per_seed.values()], ddof=1)))
               for key in ['mae', 'rmse', 'user_macro_mae']}
        models[name] = dict(per_seed=per_seed, aggregate=agg)
    for name in HEADS:
        models[name]['comparisons'] = {n:paired_seed_mean(predictions[name], ref, y, users, cfg)
                                      for n, ref in {**{n:p[None, :] for n, p in baselines.items()},
                                                     'LIST_DOT':predictions['DOT']}.items()}
    result = dict(status='PASS', protocol=cfg, baselines={n:metrics(p, y, users) for n, p in baselines.items()},
                  models=models, method_selection=read(root/'method_selection.json'))
    save(root/'summary.json', result)
    with (root/'per_seed.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    aggregate_rows = [dict(head=n, mae_mean=v['aggregate']['mae']['mean'], mae_std=v['aggregate']['mae']['std'],
                           rmse_mean=v['aggregate']['rmse']['mean'], rmse_std=v['aggregate']['rmse']['std'])
                      for n, v in models.items()]
    with (root/'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(aggregate_rows[0])); w.writeheader(); w.writerows(aggregate_rows)
    lines = ['# 固定 MF Top-100：LIST 目标评分方法对照', '',
             'Book→Movie；固定一个源 MF；学生种子 2020/2021/2022；均值 ± 样本标准差；不是模型集成。', '',
             '| 方法 | MAE ↓ | RMSE ↓ |', '| --- | ---: | ---: |']
    for n, v in result['baselines'].items(): lines.append(f"| {n} | {v['mae']:.6f} | {v['rmse']:.6f} |")
    for r in aggregate_rows:
        lines.append(f"| LIST {r['head']} | {r['mae_mean']:.6f} ± {r['mae_std']:.6f} | {r['rmse_mean']:.6f} ± {r['rmse_std']:.6f} |")
    lines += ['', '测试前按三种子平均验证 RMSE 选定的方法：'+result['method_selection']['selected_head']+'。',
              '每次均训练 10 轮，以验证 RMSE 选 checkpoint；不根据测试集选 epoch、学习率或方法。',
              '共享列表编码器结构、初始公共参数、训练批次、dropout 种子和均值/物品偏置。编码器随各自模型共同训练。',
              'GMF 从全 1 权重开始，DNN 从单位变换开始，内积加 MLP 从零残差开始；纯 MLP 默认随机初始化。',
              '所有无列表用户的交互项在评分后强制为零；未见目标物品回退既有平滑均值。',
              '原种子 2020 内积结果与上一轮 MF-list LIST 回放核验通过。全部 15 次运行冻结后读取测试标签。',
              '本轮固定同一 MF 源列表，三个种子仅反映学生训练随机性；并非源模型重训或跨数据集验证。',
              '随机用户划分、历史测试集已使用；探索性结果，用户簇区间未校正多重比较。',
              '细节见 protocol.json、summary.json、per_seed.csv、verification.json。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('LIST_HEAD_STUDY_PASS', json.dumps(aggregate_rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'selftest', 'train', 'verify', 'evaluate'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--head', choices=HEADS)
    a = p.parse_args()
    if a.stage == 'train':
        assert a.seed is not None and a.head is not None
        train(a.root.resolve(), a.seed, a.head)
    else:
        globals()[a.stage](a.root.resolve())
