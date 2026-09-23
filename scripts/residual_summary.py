"""One final held-out evaluation after all task/seed selections are frozen."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from book_movie_protocol import BASE, digest, save, predict

TASKS = ['book_movie', 'movie_book', 'movie_music_clean']


def metrics(pred, y, users):
    err = pred-y
    count = np.bincount(users)
    return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt(np.square(err).mean())),
                user_macro_mae=float((np.bincount(users, weights=np.abs(err))/count).mean()),
                prediction_mean=float(pred.mean()), prediction_std=float(pred.std()),
                outside_rating_scale_fraction=float(((pred < 1)|(pred > 5)).mean()))


def paired_interval(a, b, y, users, cfg):
    """User-cluster bootstrap of losses averaged over the fixed training seeds."""
    abs_a, abs_b = np.abs(a-y).mean(0), np.abs(b-y).mean(0)
    sq_a, sq_b = np.square(a-y).mean(0), np.square(b-y).mean(0)
    counts = np.bincount(users)
    sums = [np.bincount(users, weights=v) for v in [abs_a, abs_b, sq_a, sq_b]]
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    draws = []
    for _ in range(cfg['bootstrap_repeats']):
        idx = rng.integers(len(counts), size=len(counts))
        denom = counts[idx].sum()
        aa, ab, sa, sb = [s[idx].sum()/denom for s in sums]
        draws.append([aa-ab, np.sqrt(sa)-np.sqrt(sb)])
    ci = np.quantile(draws, [.025, .975], axis=0)
    return dict(delta_mae=float(abs_a.mean()-abs_b.mean()),
                delta_rmse=float(np.sqrt(sq_a.mean())-np.sqrt(sq_b.mean())),
                mae_95ci=ci[:, 0].tolist(), rmse_95ci=ci[:, 1].tolist(),
                interpretation='negative favors real; test-user cluster uncertainty conditional on these three seeds; not a prediction ensemble')


def main(root):
    cfg = json.loads((root/'experiment.json').read_text())
    frozen = {}
    for name in TASKS:
        for seed in cfg['seeds']:
            p = root/name/f'runs/seed_{seed}/selection.json'
            selection = json.loads(p.read_text())
            assert all(selection['checks'].values())
            for arm in cfg['arms']:
                assert digest(p.parent/f'{arm}_test_predictions.npy') == selection[arm]['predictions_sha256']
                assert digest(p.parent/f'{arm}.pt') == selection[arm]['checkpoint_sha256']
            frozen[f'{name}/{seed}'] = digest(p)
    with (root/'evaluation_started.json').open('x') as stream:
        json.dump(frozen, stream, indent=2)
    results = {}
    for name in TASKS:
        task = root/name
        manifest = json.loads((task/'manifest.json').read_text())
        for p in ['data/test.csv', 'data/test_features.npy']:
            assert digest(task/p) == manifest['hashes'][p]
        frame = pd.read_csv(task/'data/test.csv', header=None, names=['uid', 'iid', 'y'])
        np.testing.assert_array_equal(frame[['uid', 'iid']].to_numpy(), np.load(task/'data/test_features.npy'))
        y = frame.y.to_numpy()
        _, users = np.unique(frame.uid, return_inverse=True)
        stats = dict(np.load(task/'baseline/statistics.npz'))
        chosen = json.loads((task/'baseline/selection.json').read_text())['selected']
        base = np.load(task/'baseline/values.npy')[frame.iid.to_numpy()]
        # The student's float32 baseline is the exact shared reference.
        baselines = dict(fixed_baseline=metrics(base, y, users),
                         item_mean=metrics(predict(stats, frame.iid.to_numpy(), 'item_mean'), y, users),
                         item_median=metrics(predict(stats, frame.iid.to_numpy(), 'item_median'), y, users))
        arms, per_seed, epochs = {}, {}, {}
        for arm in cfg['arms']:
            pred = np.stack([np.load(task/f'runs/seed_{s}/{arm}_test_predictions.npy') for s in cfg['seeds']])
            assert pred.shape == (len(cfg['seeds']), len(y)) and np.isfinite(pred).all()
            arms[arm] = pred
            per_seed[arm] = {str(s): metrics(pred[i], y, users) for i, s in enumerate(cfg['seeds'])}
            epochs[arm] = {str(s): json.loads((task/f'runs/seed_{s}/selection.json').read_text())[arm]['epoch'] for s in cfg['seeds']}
        aggregate = {arm: {key: dict(mean=float(np.mean([m[key] for m in values.values()])),
                                    std=float(np.std([m[key] for m in values.values()], ddof=1)))
                           for key in ['mae', 'rmse', 'user_macro_mae']} for arm, values in per_seed.items()}
        comparisons = {key: paired_interval(arms['real'], other, y, users, cfg)
                       for key, other in [('real_minus_baseline', base[None, :]),
                                          ('real_minus_shuffled', arms['shuffled']),
                                          ('real_minus_target_bias', arms['target_bias'])]}
        support = all(comparisons[k]['rmse_95ci'][1] < 0 for k in ['real_minus_baseline', 'real_minus_shuffled', 'real_minus_target_bias'])
        result = dict(task=name, selected_baseline=chosen, test_rows=len(y), test_users=len(np.unique(users)),
                      cleanup=manifest['cleanup'], baselines=baselines, per_seed=per_seed,
                      selected_epochs=epochs, aggregate=aggregate, comparisons=comparisons,
                      primary_rmse_source_increment_supported=support,
                      inference_scope='One fixed split, three training/permutation seeds, user-cluster CIs conditional on fitted runs')
        save(task/'results.json', result)
        results[name] = result
        print('TASK_RESULT', name, json.dumps(result), flush=True)
    save(root/'summary.json', dict(status='PASS', config=cfg, tasks=results))
    lines = ['# 三组跨域残差对照实验', '', '状态：PASS。所有训练完成、验证选择冻结后，统一读取测试标签评价。', '',
             '每组任务一个固定 60/20/20 用户划分、三个训练种子；结果为三次单模型指标的均值及样本标准差，不是预测集成。',
             '主指标为 RMSE；MAE 同时报告。残差模型以验证 RMSE 选择 epoch，包含零修正的 epoch 0。', '',
             '| 任务 | 固定基线 RMSE | 目标偏置 RMSE | 真实源 RMSE | 打乱源 RMSE | 真实源 MAE |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, r in results.items():
        def cell(arm, key='rmse'):
            v = r['aggregate'][arm][key]
            return f"{v['mean']:.6f} ± {v['std']:.6f}"
        lines.append(f"| {name} | {r['baselines']['fixed_baseline']['rmse']:.6f} | {cell('target_bias')} | {cell('real')} | {cell('shuffled')} | {cell('real', 'mae')} |")
    for name, r in results.items():
        lines += ['', f'## {name}', '', '清洗记录：`'+json.dumps(r['cleanup'], ensure_ascii=False)+'`', '',
                  '验证集选定基线：`'+r['selected_baseline']+'`。选定 epoch：`'+json.dumps(r['selected_epochs'])+'`。', '',
                  '| 差异（负数有利于真实源） | ΔRMSE | 用户簇 bootstrap 95% CI |', '| --- | ---: | --- |']
        for key, v in r['comparisons'].items():
            lines.append(f"| {key} | {v['delta_rmse']:.6f} | [{v['rmse_95ci'][0]:.6f}, {v['rmse_95ci'][1]:.6f}] |")
        lines += ['', '本轮主指标上的源域增量证据：'+('满足预设三项比较条件。' if r['primary_rmse_source_increment_supported'] else '未同时满足优于固定基线、打乱源和目标偏置的条件。')]
    lines += ['', '## 范围与复核', '',
              'Book→Movie 复用 8564 划分；其他任务采用稳定原始 ID 排序后固定用户划分。',
              'Movie→Music 在划分之前删除源电影域所有与音乐域共享 ASIN 的记录，保留音乐目标域；处理后物品交集和跨域同用户物品评分重叠均为零。',
              '清洗后没有重新做 5-core。该规则改变源域目录、历史量和重叠用户集合，清洗前后指标不能直接用于归因。',
              '全部任务仍为随机用户切分，没有时间过滤；完整目录 ID 编码和源查询包含冷用户，不是严格未知目录的归纳协议。',
              '学生只接收源 MF Top-100 ID 列表；每种子重新训练源 MF，共享给配对真实/打乱学生。训练进程禁止读取测试评分和源模型权重。',
              'CI 是固定三个训练种子下的测试用户抽样不确定性；种子稳定性另见标准差。多任务比较未做多重检验校正，不作普遍性结论。',
              '中位数是 MAE 参照，不能根据测试结果改选目标或 epoch。细分指标和全部种子结果见各任务 results.json。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('ALL_EXPERIMENTS_PASS', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
