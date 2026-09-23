"""Bounded privileged-source diagnosis on the fixed, cleaned residual protocols."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from book_movie_protocol import BASE, digest, save
from residual_experiment import seed_all, guard_test_labels, load_frame, infer, state_hash
from residual_summary import metrics, paired_interval

PRIOR = BASE/'runs/residual_study_20260922'
TASKS = ['book_movie', 'movie_book', 'movie_music_clean']
REPS = ['mf_embedding', 'history_stats']


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((PRIOR/'experiment.json').read_text())
    cfg.update(prior_root=str(PRIOR), representations=REPS, screening_seed=2020,
               arms=['real', 'shuffled'], source_models_retrained=False,
               source_api='Privileged source MF vectors or source history statistics; Top100 is a prior matched-seed reference',
               expansion_seeds=[2021, 2022], minimum_validation_gain=.0001,
               expansion_rule='Real representation improves fixed-baseline validation RMSE by at least 0.0001 in seed2020',
               architecture='fixed standardized source features padded to 32 -> 64 -> 32, dot target embedding, plus item bias',
               information_scope='Privileged diagnosis, not a black-box method or a mathematical upper bound',
               normalization='mean/std from training overlap users only; applied before within-split permutation',
               code_sha256={name: digest(BASE/'scripts'/name) for name in
                            ['representation_diagnostic.py', 'residual_experiment.py', 'residual_summary.py', 'book_movie_protocol.py']})
    save(root/'protocol.json', cfg)
    for taskname in TASKS:
        old, task = PRIOR/taskname, root/taskname
        task.mkdir()
        (task/'features').mkdir()
        manifest = json.loads((old/'manifest.json').read_text())
        (task/'data').symlink_to(old/'data', target_is_directory=True)
        (task/'baseline').symlink_to(old/'baseline', target_is_directory=True)
        save(task/'manifest.json', manifest)
        for name in ['data/source.csv', 'data/train_users.npy', 'data/validation_users.npy', 'data/test_users.npy']:
            assert digest(old/name) == manifest['hashes'][name]
        src = load_frame(old, 'source')
        n = manifest['overlap_users']
        group = src[src.uid < n]
        uid = group.uid.to_numpy(dtype=np.int64)
        y = group.y.to_numpy(dtype=float)
        count = np.bincount(uid, minlength=n)
        assert np.all(count > 0)
        mean = np.bincount(uid, weights=y, minlength=n)/count
        variance = np.maximum(np.bincount(uid, weights=y*y, minlength=n)/count-mean*mean, 0)
        hist = np.bincount(uid*5+y.astype(int)-1, minlength=n*5).reshape(n, 5)/count[:, None]
        item_means = src.groupby('iid').y.mean()
        centered = y-group.iid.map(item_means).to_numpy()
        item_count = src.groupby('iid').size()
        popularity = np.log1p(group.iid.map(item_count).to_numpy())
        raw = np.column_stack([np.log1p(count), mean, np.sqrt(variance), hist,
                               np.bincount(uid, weights=centered, minlength=n)/count,
                               np.bincount(uid, weights=popularity, minlength=n)/count])
        columns = ['log_count', 'rating_mean', 'rating_std']+[f'fraction_{k}_stars' for k in range(1, 6)]+['item_centered_mean', 'mean_log_item_count']
        train_users = np.load(old/'data/train_users.npy')
        records = {}
        def export(raw, label, inputs, columns):
            center, scale = raw[train_users].mean(0), raw[train_users].std(0)
            scale = np.maximum(scale, 1e-6)
            standardized = (raw-center)/scale
            features = np.pad(standardized, ((0, 0), (0, 32-standardized.shape[1]))).astype(np.float32)
            assert np.isfinite(features).all()
            path = task/'features'/f'{label}.npz'
            np.savez(path, features=features, raw=raw, center=center, scale=scale)
            records[label] = dict(columns=columns, inputs=inputs, sha256=digest(path),
                                  normalization_train_users=len(train_users), target_labels_read=False)
        export(raw, 'history_stats', {'source_csv_sha256': manifest['hashes']['data/source.csv']}, columns)
        for seed in cfg['seeds']:
            src_out = old/f'runs/seed_{seed}'
            complete = json.loads((src_out/'source_complete.json').read_text())
            assert digest(src_out/'source.pt') == complete['source_sha256']
            assert complete['source_data_sha256'] == manifest['hashes']['data/source.csv']
            state = torch.load(src_out/'source.pt', map_location='cpu', weights_only=True)
            raw = state['users.weight'][:n].numpy().copy()
            assert raw.shape == (n, 10)
            export(raw, f'mf_embedding_{seed}', {'source_checkpoint_sha256': complete['source_sha256']},
                   [f'source_mf_dim_{i}' for i in range(10)])
        save(task/'features/manifest.json', records)
        print('FEATURES_READY', taskname, n, flush=True)


class FeatureResidual(nn.Module):
    def __init__(self, values, seen, features, cfg):
        super().__init__()
        self.register_buffer('baseline', values)
        self.register_buffer('seen', seen)
        self.register_buffer('features', features, persistent=False)
        self.bias = nn.Embedding(len(values), 1)
        self.target_items = nn.Embedding(len(values), 32)
        self.encoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Dropout(cfg['student_dropout']), nn.Linear(64, 32))
        nn.init.zeros_(self.bias.weight)
        nn.init.normal_(self.target_items.weight, std=.1)
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, x):
        u, i = x[:, 0], x[:, 1]
        z = self.encoder(self.features[u.clamp(max=len(self.features)-1)])*(u < len(self.features))[:, None]
        delta = self.bias(i).squeeze(1)+(z*self.target_items(i)).sum(1)
        return self.baseline[i]+delta*self.seen[i]


def selftest():
    seed_all(1)
    values = torch.tensor([3., 4., 2.], device='cuda')
    seen = torch.tensor([True, True, False], device='cuda')
    f = torch.randn(2, 32, device='cuda')
    seed_all(2)
    model = FeatureResidual(values, seen, f, {'student_dropout': .1}).cuda()
    seed_all(2)
    other = FeatureResidual(values, seen, f.flip(0), {'student_dropout': .1}).cuda()
    assert state_hash(model) == state_hash(other)
    x = torch.tensor([[0, 0], [1, 1], [0, 2], [3, 0]], device='cuda')
    torch.testing.assert_close(model(x), values[x[:, 1]], rtol=0, atol=0)
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(3):
        opt.zero_grad()
        (model(x)-1).square().mean().backward()
        opt.step()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.encoder[-1].weight.abs().sum() > 0
    assert model(x)[2] == values[2]
    print('DIAGNOSTIC_SELFTEST_PASS', flush=True)


def train(root, index, seed):
    cfg = json.loads((root/'protocol.json').read_text())
    for name, expected in cfg['code_sha256'].items():
        assert digest(BASE/'scripts'/name) == expected
    taskname, rep = TASKS[index//2], REPS[index%2]
    if seed != cfg['screening_seed']:
        screen = json.loads((root/'screening.json').read_text())
        if not screen[f'{taskname}/{rep}']['expand']:
            print('SKIP_NO_VALIDATION_GAIN', taskname, rep, seed, flush=True)
            return
    task, prior = root/taskname, PRIOR/taskname
    guard_test_labels(task, student=True)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    out = task/rep/f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((task/'manifest.json').read_text())
    train_df, val = load_frame(task, 'train'), load_frame(task, 'validation')
    assert not set(train_df.uid) & set(val.uid)
    fm = json.loads((task/'features/manifest.json').read_text())
    label = rep if rep == 'history_stats' else f'{rep}_{seed}'
    fp = task/'features'/f'{label}.npz'
    assert digest(fp) == fm[label]['sha256']
    features = np.load(fp)['features']
    pp = prior/f'runs/seed_{seed}/permutation.npy'
    permutation = np.load(pp)
    for g in ['train', 'validation', 'test']:
        users = np.load(task/f'data/{g}_users.npy')
        np.testing.assert_array_equal(np.sort(permutation[users]), np.sort(users))
        assert np.all(permutation[users] != users)
    values = torch.tensor(np.load(task/'baseline/values.npy'), device='cuda')
    seen = torch.tensor(np.load(task/'baseline/seen.npy'), device='cuda')
    x = torch.tensor(train_df[['uid', 'iid']].to_numpy(), dtype=torch.long, device='cuda')
    y = torch.tensor(train_df.y.to_numpy(), dtype=torch.float32, device='cuda')
    vx = torch.tensor(val[['uid', 'iid']].to_numpy(), dtype=torch.long, device='cuda')
    frozen, initial_hash = {}, None
    for arm in ['real', 'shuffled']:
        seed_all(seed+200)
        f = torch.tensor(features if arm == 'real' else features[permutation], device='cuda')
        model = FeatureResidual(values, seen, f, cfg).cuda()
        init = state_hash(model)
        if arm == 'real': initial_hash = init
        else: assert init == initial_hash
        p = infer(model, vx)
        np.testing.assert_array_equal(p, values[vx[:, 1]].cpu().numpy())
        best = float(np.sqrt(np.square(p-val.y.to_numpy()).mean()))
        chosen, history = 0, [dict(epoch=0, validation_rmse=best)]
        checkpoint = out/f'{arm}.pt'
        torch.save(model.state_dict(), checkpoint)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['student_lr'], weight_decay=cfg['student_weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(seed+300)
        for epoch in range(1, cfg['student_epochs']+1):
            tick = time.monotonic()
            model.train()
            total = torch.zeros((), device='cuda')
            for idx in torch.randperm(len(x), device='cuda', generator=generator).split(cfg['student_batch']):
                opt.zero_grad(set_to_none=True)
                loss = (model(x[idx])-y[idx]).square().mean()
                loss.backward()
                opt.step()
                total += loss.detach()*len(idx)
            p = infer(model, vx)
            rmse = float(np.sqrt(np.square(p-val.y.to_numpy()).mean()))
            assert np.isfinite(rmse) and torch.isfinite(total)
            if rmse < best:
                best, chosen = rmse, epoch
                torch.save(model.state_dict(), checkpoint)
            row = dict(epoch=epoch, train_mse=total.item()/len(x), validation_rmse=rmse,
                       validation_mae=float(np.abs(p-val.y.to_numpy()).mean()), seconds=time.monotonic()-tick)
            history.append(row)
            save(out/f'{arm}_training.json', history)
            print('DIAG_EPOCH', taskname, rep, seed, arm, json.dumps(row), flush=True)
        model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
        p = infer(model, vx)
        assert abs(float(np.sqrt(np.square(p-val.y.to_numpy()).mean()))-best) < 1e-12
        np.save(out/f'{arm}_validation.npy', p)
        info = dict(epoch=chosen, validation_rmse=best, initialization_sha256=init,
                    checkpoint_sha256=digest(checkpoint), feature_sha256=digest(fp),
                    permutation_sha256=digest(pp), test_labels_read=False)
        save(out/f'{arm}_selection.json', info)
        test_path = task/'data/test_features.npy'
        assert digest(test_path) == manifest['hashes']['data/test_features.npy']
        tx = torch.tensor(np.load(test_path), dtype=torch.long, device='cuda')
        pred = infer(model, tx)
        assert pred.shape == (len(tx),) and np.isfinite(pred).all()
        np.testing.assert_array_equal(pred[~seen[tx[:, 1]].cpu().numpy()], values[tx[:, 1]][~seen[tx[:, 1]]].cpu().numpy())
        np.save(out/f'{arm}_test.npy', pred)
        info['prediction_sha256'] = digest(out/f'{arm}_test.npy')
        frozen[arm] = info
        del opt, model, f, tx
        torch.cuda.empty_cache()
    save(out/'selection.json', frozen)
    print('REPRESENTATION_FROZEN', taskname, rep, seed, flush=True)


def screen(root):
    cfg = json.loads((root/'protocol.json').read_text())
    assert not (root/'screening.json').exists()
    rows = {}
    for name in TASKS:
        for rep in REPS:
            out = root/name/rep/f"seed_{cfg['screening_seed']}"
            sel = json.loads((out/'selection.json').read_text())
            base = json.loads((out/'real_training.json').read_text())[0]['validation_rmse']
            gain = base-sel['real']['validation_rmse']
            rows[f'{name}/{rep}'] = dict(baseline_validation_rmse=base, real_validation_rmse=sel['real']['validation_rmse'],
                                       shuffled_validation_rmse=sel['shuffled']['validation_rmse'], gain=gain,
                                       expand=bool(gain >= cfg['minimum_validation_gain']),
                                       seeds=cfg['seeds'] if gain >= cfg['minimum_validation_gain'] else [cfg['screening_seed']])
    save(root/'screening.json', rows)
    print('SCREENING_DECISIONS_FROZEN', json.dumps(rows), flush=True)


def evaluate(root):
    cfg = json.loads((root/'protocol.json').read_text())
    screening = json.loads((root/'screening.json').read_text())
    frozen = {}
    for key, rule in screening.items():
        for seed in rule['seeds']:
            p = root/key/f'seed_{seed}/selection.json'
            sel = json.loads(p.read_text())
            assert sel['real']['initialization_sha256'] == sel['shuffled']['initialization_sha256']
            for arm in ['real', 'shuffled']:
                assert digest(p.parent/f'{arm}.pt') == sel[arm]['checkpoint_sha256']
                assert digest(p.parent/f'{arm}_test.npy') == sel[arm]['prediction_sha256']
            frozen[f'{key}/{seed}'] = digest(p)
    with (root/'evaluation_started.json').open('x') as f:
        json.dump(frozen, f, indent=2)
    results, verification = {}, []
    for taskname in TASKS:
        task = root/taskname
        manifest = json.loads((task/'manifest.json').read_text())
        for rel, expected in manifest['hashes'].items():
            assert digest(task/rel) == expected
        for rel, expected in manifest['baseline_hashes'].items():
            assert digest(task/rel) == expected
        test = pd.read_csv(task/'data/test.csv', header=None, names=['uid', 'iid', 'y'])
        np.testing.assert_array_equal(test[['uid', 'iid']].to_numpy(), np.load(task/'data/test_features.npy'))
        val = pd.read_csv(task/'data/validation.csv', header=None).to_numpy()
        y = test.y.to_numpy()
        _, users = np.unique(test.uid, return_inverse=True)
        base = np.load(task/'baseline/values.npy')[test.iid.to_numpy()]
        task_result = dict(baseline=metrics(base, y, users), representations={})
        for rep in REPS:
            rule = screening[f'{taskname}/{rep}']
            arrays, by_seed, epochs = {}, {}, {}
            for arm in ['real', 'shuffled']:
                rows = []
                by_seed[arm], epochs[arm] = {}, {}
                for seed in rule['seeds']:
                    out = task/rep/f'seed_{seed}'
                    sel = json.loads((out/'selection.json').read_text())[arm]
                    history = json.loads((out/f'{arm}_training.json').read_text())
                    assert [h['epoch'] for h in history] == list(range(cfg['student_epochs']+1))
                    assert min(history, key=lambda h: h['validation_rmse'])['epoch'] == sel['epoch']
                    pv = np.load(out/f'{arm}_validation.npy')
                    assert abs(float(np.sqrt(np.square(pv-val[:, 2]).mean()))-sel['validation_rmse']) < 1e-12
                    verification.append(dict(task=taskname, representation=rep, seed=seed, arm=arm, passed=True))
                    pred = np.load(out/f'{arm}_test.npy')
                    rows.append(pred)
                    by_seed[arm][str(seed)] = metrics(pred, y, users)
                    epochs[arm][str(seed)] = sel['epoch']
                arrays[arm] = np.stack(rows)
            refs = {a: np.stack([np.load(PRIOR/taskname/f'runs/seed_{s}/{a}_test_predictions.npy') for s in rule['seeds']])
                    for a in ['real', 'target_bias']}
            comparisons = {label: paired_interval(arrays['real'], ref, y, users, cfg) for label, ref in
                           [('minus_baseline', base[None, :]), ('minus_shuffled', arrays['shuffled']),
                            ('minus_top100', refs['real']), ('minus_target_bias', refs['target_bias'])]}
            for v in comparisons.values():
                v['interpretation'] = f"negative favors this representation; user-cluster CI conditional on {len(rule['seeds'])} fitted seeds"
            aggregate = {a: {m: dict(mean=float(np.mean([r[m] for r in records.values()])),
                                    std=float(np.std([r[m] for r in records.values()], ddof=1)) if len(records)>1 else None)
                             for m in ['mae', 'rmse', 'user_macro_mae']} for a, records in by_seed.items()}
            task_result['representations'][rep] = dict(seeds=rule['seeds'], screening=rule, per_seed=by_seed,
                                                       aggregate=aggregate, epochs=epochs, comparisons=comparisons,
                                                       source_increment_supported=all(comparisons[k]['rmse_95ci'][1] < 0 for k in
                                                                                      ['minus_baseline', 'minus_shuffled', 'minus_target_bias']))
        results[taskname] = task_result
        save(task/'results.json', task_result)
        print('DIAGNOSTIC_RESULT', taskname, json.dumps(task_result), flush=True)
    save(root/'summary.json', dict(status='PASS', tasks=results, protocol=cfg))
    save(root/'verification.json', dict(status='PASS', models=verification, checks=['paired initialization', 'all epochs trained',
         'validation minimum selection independently checked', 'restored model validation predictions match',
         'data and fixed baseline unchanged', 'prediction and model hashes verified']))
    lines = ['# 源表示诊断结果', '', '状态：PASS。仅验证集决定补跑种子与选定 epoch，所有选择冻结后统一测试。', '',
             '| 任务 | 固定基线 RMSE | MF向量 RMSE | 历史统计 RMSE | MF/统计种子数 |',
             '| --- | ---: | ---: | ---: | --- |']
    for name, r in results.items():
        a, b = [r['representations'][rep] for rep in REPS]
        lines.append(f"| {name} | {r['baseline']['rmse']:.6f} | {a['aggregate']['real']['rmse']['mean']:.6f} | {b['aggregate']['real']['rmse']['mean']:.6f} | {len(a['seeds'])}/{len(b['seeds'])} |")
    for name, r in results.items():
        lines += ['', f'## {name}', '']
        for rep, d in r['representations'].items():
            lines += [f'### {rep}', '', f"种子：{d['seeds']}；选定 epoch：{d['epochs']}。", '',
                      '| 真实表示相对 | ΔRMSE | 95% CI |', '| --- | ---: | --- |']
            for name2, c in d['comparisons'].items():
                lines.append(f"| {name2} | {c['delta_rmse']:.6f} | [{c['rmse_95ci'][0]:.6f}, {c['rmse_95ci'][1]:.6f}] |")
            lines += ['', '本轮满足预设源增量条件：'+str(d['source_increment_supported'])+'。']
    lines += ['', '## 范围', '', '本轮是白盒/源历史特权诊断，不是黑盒方法，也不是严格信息上界。',
              '复用上一轮同一源 MF 权重和数据，控制源服务变化；历史统计只包含评分风格和活跃度，不包含完整语义偏好。',
              'MF/统计表示使用相同 32 维输入、相同模型结构和超参数；相对 Top100，输入前端与可训练参数量不同，因此不能仅归因为信息量。',
              '特征标准化只拟合训练重叠用户；测试和验证源历史可用于各自特征，但目标评分仅用于对应评价。',
              '一个固定划分；先筛查后补种子，区间条件于实际运行种子，未做多重检验校正。',
              '历史测试集此前已使用，本轮是探索性诊断，不能作为新盲测确认。',
              '详细 MAE、种子差异、验证筛查及校验见 summary.json、screening.json、verification.json。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('REPRESENTATION_DIAGNOSIS_PASS', len(verification), 'models', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'selftest', 'train', 'screen', 'evaluate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--seed', type=int, default=2020)
    args = parser.parse_args()
    if args.stage == 'train': train(args.root, args.index, args.seed)
    elif args.stage == 'selftest': selftest()
    else: globals()[args.stage](args.root)
