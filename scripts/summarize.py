"""Report fixed-epoch metrics and paired cold-user bootstrap uncertainty."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
results = ROOT / 'results'
data = {key: np.load(results / f'{key}_predictions.npz') for key in ['tgt', 'wb', 'bb']}
for key in ['wb', 'bb']:
    for column in ['uid', 'iid', 'y']:
        np.testing.assert_array_equal(data['tgt'][column], data[key][column])
errors = {key: np.abs(value['y'] - value['prediction']) for key, value in data.items()}
mae = {key: float(value.mean()) for key, value in errors.items()}
gain_wb = mae['tgt'] - mae['wb']
gain_bb = mae['tgt'] - mae['bb']
gap = mae['bb'] - mae['wb']
ratio = gain_bb / gain_wb if gain_wb > 1e-8 else None
users, inverse = np.unique(data['tgt']['uid'], return_inverse=True)
counts = np.bincount(inverse)
sums = {key: np.bincount(inverse, weights=value) for key, value in errors.items()}
rng = np.random.default_rng(2020)
boot = {key: [] for key in ['gain_wb', 'gain_bb', 'gap']}
for _ in range(2000):
    selected = rng.integers(0, len(users), len(users))
    means = {key: float(value[selected].sum() / counts[selected].sum()) for key, value in sums.items()}
    boot['gain_wb'].append(means['tgt'] - means['wb'])
    boot['gain_bb'].append(means['tgt'] - means['bb'])
    boot['gap'].append(means['bb'] - means['wb'])
ci = {key: np.quantile(value, [.025, .975]).tolist() for key, value in boot.items()}
summary = {'mae': mae, 'gain_wb': gain_wb, 'gain_bb': gain_bb, 'gap': gap, 'recovery_ratio': ratio,
           'paired_user_bootstrap_95ci': ci, 'seed': 2020, 'epoch': 10,
           'scope': 'single split and training seed; bootstrap excludes retraining uncertainty',
           'bb_method': 'plain Top-100 + score distillation; NOT DBGRME'}
(results / 'summary.json').write_text(json.dumps(summary, indent=2))
with (results / 'metrics.csv').open('w') as file:
    writer = csv.writer(file)
    writer.writerow(['setting', 'task', 'backbone', 'beta', 'seed', 'mae', 'run_id'])
    for key, name in [('tgt', 'TGT'), ('wb', 'PFCDR-WB'), ('bb', 'PFCDR-BB-ScoreDistill')]:
        writer.writerow([name, 'Book->Movie', 'MF', .2, 2020, mae[key], 'seed2020_final_epoch10'])
def interval(key):
    return f"[{ci[key][0]:.6f}, {ci[key][1]:.6f}]"
def answer(key):
    return '本次实验支持正收益' if ci[key][0] > 0 else '本次实验未确认稳定的正收益'
lines = [
    '# Baby Check：第一轮结果', '',
    'Book→Movie；MF；β=20%；seed=2020；固定第10轮。', '',
    '| Setting | MAE ↓ |', '| --- | ---: |',
    f"| TGT | {mae['tgt']:.6f} |", f"| PFCDR-WB | {mae['wb']:.6f} |",
    f"| PFCDR-BB（普通分数蒸馏） | {mae['bb']:.6f} |", '',
    f'G_WB={gain_wb:.6f}；G_BB={gain_bb:.6f}；BB−WB={gap:.6f}；RecoveryRatio={ratio}.', '',
    f"Q1：{answer('gain_wb')}；配对用户 bootstrap 95% CI：{interval('gain_wb')}。", '',
    f"Q2：{'本次黑盒基线明显劣于白盒' if ci['gap'][0] > 0 else '未确认黑盒基线明显劣于白盒'}；差距95% CI：{interval('gap')}。", '',
    f"Q3：{answer('gain_bb')}；95% CI：{interval('gain_bb')}。", '',
    '限制：只有一个划分与训练种子，区间不包含重训练不确定性。黑盒组不是 DBGRME，不能将其差距解释为所有黑盒方法的固有上限。', '',
    '官方 TGT 对完全冷启动用户使用未训练的随机用户向量；因此相对该基线的收益不能单独证明源知识优于合理的目标域冷启动回退策略。', '',
    '实现及兼容性差异见 ../docs/reproduction_notes.md。逐轮测试指标保存在 base_training.json、wb_training.json、bb_training.json，不据此选择主报告轮次。',
]
(results / 'report.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(summary, indent=2))
