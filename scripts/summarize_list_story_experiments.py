"""Independently audit and compare frozen IB/acquisition/memory LIST runs on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PARENT = BASE / 'runs/small_task_comparison_8650'
SEEDS = [2020, 2021, 2022]
BETAS = [20, 80]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metric(pred, y):
    err = pred - y
    return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt(np.square(err).mean())))


def audit_and_summarize(tracks, out):
    # Do not open labels until every track has finished and frozen its predictions.
    for name, root in tracks.items():
        assert read(root / 'summary.json')['status'] == 'PASS', name
        assert read(root / 'verification.json')['status'] == 'PASS', name
        assert (root / 'evaluation_started.json').exists(), name
    out.mkdir(parents=True, exist_ok=False)
    data = {}
    parent_rows = []
    for beta in BETAS:
        cell = PARENT / f'beta_{beta}'
        manifest = read(cell / 'manifest.json')
        for name, value in manifest['hashes'].items():
            assert digest(cell / name) == value, name
        frames = {name: pd.read_csv(cell / f'data/{name}.csv', header=None,
                                   names=['u', 'i', 'y'])
                  for name in ['train', 'inner_train', 'validation', 'test']}
        train, inner, val, test = [frames[n] for n in ['train', 'inner_train', 'validation', 'test']]
        assert not set(train.u) & set(test.u)
        assert not set(inner.u) & set(val.u)
        assert not set(val.u) & set(test.u)
        assert set(inner.u) | set(val.u) <= set(train.u)
        np.testing.assert_array_equal(test[['u', 'i']], np.load(cell / 'data/test_features.npy'))
        val_base = val.i.map(inner.groupby('i').y.mean()).fillna(inner.y.mean()).to_numpy()
        base = test.i.map(train.groupby('i').y.mean()).fillna(train.y.mean()).to_numpy()
        y = test.y.to_numpy()
        refs = {}
        for seed in SEEDS:
            old = cell / f'seed_{seed}/residual_real'
            done = read(old / 'complete.json')
            assert digest(old / 'test.npy') == done['test_sha256']
            refs[seed] = metric(np.load(old / 'test.npy'), y)
            parent_rows.append(dict(track='reference', variant='LIST_v1', beta=beta,
                                    seed=seed, **refs[seed]))
            parent_rows.append(dict(track='reference', variant='ordinary_mean', beta=beta,
                                    seed=seed, **metric(base, y)))
        data[beta] = dict(y=y, val_y=val.y.to_numpy(), val_base=val_base, base=base,
                          unseen=~test.i.isin(set(train.i)).to_numpy(), train_rows=len(train), refs=refs)
    rows = []
    evidence = []
    for track, root in tracks.items():
        cfg = read(root / 'protocol.json')
        for filename, value in cfg['code_hashes'].items():
            assert digest(Path(filename)) == value, filename
        evaluation_time = (root / 'evaluation_started.json').stat().st_mtime_ns
        declared = pd.read_csv(root / 'per_seed.csv')
        reference_names = {'item_mean', 'LIST_original', 'LIST_v1', 'LIST_residual_real',
                           'ordinary_mean', 'original_LIST', 'k_100_original'}
        missing_dirs = declared.variant.map(lambda v: not (root / v).is_dir())
        assert set(declared.loc[missing_dirs, 'variant']) <= reference_names
        declared = declared.loc[~missing_dirs].copy()
        assert not declared.duplicated(['variant', 'beta', 'seed']).any()
        variants = list(declared.variant.unique())
        assert len(declared) == len(variants) * len(BETAS) * len(SEEDS)
        for _, row in declared.iterrows():
            beta, seed, variant = int(row.beta), int(row.seed), row.variant
            assert beta in BETAS and seed in SEEDS
            p = root / variant / f'beta_{beta}/seed_{seed}/residual_real'
            done = read(p / 'complete.json')
            assert done['status'] == 'PASS'
            assert done['test_labels_opened'] is False
            assert done['restored_predictions'] and done['refit_reinitialized']
            assert done['full_outer_train_rows'] == data[beta]['train_rows']
            for filename, key in [('final.pt', 'final_sha256'), ('test.npy', 'test_sha256'),
                                  ('selection.json', 'selection_sha256')]:
                assert digest(p / filename) == done[key], p / filename
                assert (p / filename).stat().st_mtime_ns < evaluation_time
            assert (p / 'complete.json').stat().st_mtime_ns < evaluation_time
            history = read(p / 'select_history.json')
            assert [r['epoch'] for r in history] == list(range(1, cfg['epochs'] + 1))
            expected = dict(epoch=0, lambda_value=0.,
                            validation_rmse=metric(data[beta]['val_base'], data[beta]['val_y'])['rmse'])
            for epoch in history:
                for lam in cfg['lambdas']:
                    value = epoch['validation'][str(lam)]
                    if value < expected['validation_rmse']:
                        expected = dict(epoch=epoch['epoch'], lambda_value=lam, validation_rmse=value)
            selected = read(p / 'selection.json')
            for key, value in expected.items():
                assert abs(selected[key] - value) < 1e-12, (p, key, selected, expected)
            vp = np.load(p / 'validation.npy')
            assert abs(metric(vp, data[beta]['val_y'])['rmse'] - selected['validation_rmse']) < 1e-12
            assert len(read(p / 'refit_history.json')) == selected['epoch'] == done['refit_epochs']
            pred = np.load(p / 'test.npy')
            assert pred.shape == data[beta]['y'].shape and np.isfinite(pred).all()
            np.testing.assert_allclose(pred[data[beta]['unseen']], data[beta]['base'][data[beta]['unseen']],
                                       atol=1e-12, rtol=0)
            values = metric(pred, data[beta]['y'])
            for key in values:
                assert abs(values[key] - row[key]) < 1e-12, (p, key)
            ref = data[beta]['refs'][seed]
            rows.append(dict(track=track, variant=variant, beta=beta, seed=seed, **values,
                             delta_mae=values['mae'] - ref['mae'], delta_rmse=values['rmse'] - ref['rmse'],
                             epoch=selected['epoch'], lambda_value=selected['lambda_value']))
            evidence.append(dict(track=track, model=str(p), complete_sha256=digest(p / 'complete.json'),
                                 trainable_parameters=done.get('trainable_parameters'),
                                 epoch=selected['epoch'], lambda_value=selected['lambda_value']))
    pd.DataFrame(rows).to_csv(out / 'per_seed.csv', index=False)
    combined = pd.DataFrame(parent_rows + rows)
    aggregate = []
    for (track, variant, beta), group in combined.groupby(['track', 'variant', 'beta'], sort=False):
        assert sorted(group.seed.tolist()) == SEEDS
        ref = combined[(combined.variant == 'LIST_v1') & (combined.beta == beta)].sort_values('seed')
        group = group.sort_values('seed')
        baseline_mae, baseline_rmse = ref.mae.mean(), ref.rmse.mean()
        aggregate.append(dict(track=track, variant=variant, beta=int(beta),
                              mae=group.mae.mean(), mae_std=group.mae.std(ddof=1),
                              rmse=group.rmse.mean(), rmse_std=group.rmse.std(ddof=1),
                              delta_mae=group.mae.mean() - baseline_mae,
                              delta_rmse=group.rmse.mean() - baseline_rmse,
                              relative_rmse_change_pct=100 * (group.rmse.mean() / baseline_rmse - 1),
                              rmse_seed_wins=int((group.rmse.to_numpy() < ref.rmse.to_numpy()).sum())))
    table = pd.DataFrame(aggregate)
    table.to_csv(out / 'comparison.csv', index=False)
    audit = dict(status='PASS', new_models=len(rows), seeds=SEEDS, betas=BETAS,
                 tracks={k: str(v) for k, v in tracks.items()},
                 baseline=str(PARENT), verifier_sha256=digest(Path(__file__)),
                 checks=['all tracks completed before combined evaluation', 'frozen code and data hashes',
                         'disjoint cold-user splits', 'independently recomputed ordinary training means',
                         'prediction/checkpoint/selection hashes and freeze timestamps',
                         'validation-only epoch and lambda argmin', 'full refit for selected epochs',
                         'unseen-item fallback', 'independently recomputed per-seed MAE/RMSE'],
                 models=evidence)
    (out / 'independent_verification.json').write_text(json.dumps(audit, indent=2) + '\n')
    (out / 'summary.json').write_text(json.dumps(dict(status='PASS', rows=aggregate,
                                                      tracks=audit['tracks']), indent=2) + '\n')
    lines = ['# LIST 三条路线实验对比', '',
             '最小任务：去重 Music→Movie。β为冷用户测试比例；各格为三个独立训练种子的指标平均，越小越好。',
             '原版 LIST 指 GitHub 归档的全局排名池化＋普通均值残差版本（8650 residual_real），不是更早的非残差 LIST。',
             '各改动独立实施，没有将 IB、采集和记忆混合。所有候选结果均列出，不按测试结果重新选模。',
             'Δ为新版本减原版；负数表示改善。seed 胜出数仅是固定划分下的三个训练种子，不代表三个独立数据划分。', '',
             '| 路线 | 版本 | β | MAE ± SD | RMSE ± SD | ΔMAE | ΔRMSE | RMSE变化 | seed胜出 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in aggregate:
        lines.append(f"| {r['track']} | {r['variant']} | {r['beta']}% | {r['mae']:.6f} ± {r['mae_std']:.6f} | "
                     f"{r['rmse']:.6f} ± {r['rmse_std']:.6f} | {r['delta_mae']:+.6f} | {r['delta_rmse']:+.6f} | "
                     f"{r['relative_rmse_change_pct']:+.3f}% | {r['rmse_seed_wins']}/3 |")
    lines += ['', f'共独立复核 {len(rows)} 个新模型；核验结果见 independent_verification.json。',
              '这批结果来自已多次用于探索的历史测试划分，尚不能作为新数据上的最终确认。', '', '运行路径：', '']
    lines += [f'- {name}: `{root}`' for name, root in tracks.items()]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(table.to_string(index=False))
    print(f'ALL_PASS: {len(rows)} new models, output={out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['ib', 'acquisition', 'memory']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit_and_summarize({name: getattr(args, name).resolve() for name in ['ib', 'acquisition', 'memory']},
                        args.out.resolve())
