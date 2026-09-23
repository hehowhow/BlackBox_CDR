"""Prepare a fixed cold-user validation protocol, select baselines, evaluate once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OLD = BASE / 'data/ready/_8_2/tgt_Movies_and_TV_src_Books'
ALPHAS = [1, 5, 10, 20, 50, 100]
NAMES = ['global_mean', 'global_median', 'item_mean', 'item_median'] + [f'smoothed_mean_{a}' for a in ALPHAS]


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return pd.read_csv(path, header=None, names=['uid', 'iid', 'y'])


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for name in ['data', 'data/id_maps', 'results', 'models']:
        (root / name).mkdir()
    data = root / 'data'
    old_manifest = json.loads((BASE/'data/split_manifest.json').read_text())
    n = old_manifest['overlap_users']
    test_users = pd.read_csv(OLD/'test_list.csv', header=None).to_numpy().ravel().astype(int)
    remaining = sorted(set(range(n)) - set(test_users))
    val_users = sorted(random.Random(20220922).sample(remaining, round(.2*n)))
    train_users = sorted(set(remaining) - set(val_users))
    assert not set(train_users) & set(val_users)
    assert not set(train_users) & set(test_users)
    assert not set(val_users) & set(test_users)
    assert len(train_users) + len(val_users) + len(test_users) == n
    target = read(OLD/'train_tgt.csv')
    assert not set(target.uid) & set(test_users)
    val_mask = target.uid.isin(val_users)
    train, val = target.loc[~val_mask], target.loc[val_mask]
    meta = train[train.uid.isin(train_users)]
    assert set(val.uid) == set(val_users)
    assert set(train.uid) & set(range(n)) == set(train_users)
    assert not train.uid.isin(list(val_users) + list(test_users)).any()
    train.to_csv(data/'train_tgt.csv', header=False, index=False)
    val.to_csv(data/'validation.csv', header=False, index=False)
    meta.to_csv(data/'train_meta.csv', header=False, index=False)
    # Freeze the existing test file byte-for-byte; no label parsing in preparation.
    shutil.copy2(OLD/'test.csv', data/'test.csv')
    (data/'train_src.csv').symlink_to(OLD/'train_src.csv')
    for name in ['source_users', 'target_users', 'source_items', 'target_items']:
        shutil.copy2(BASE/f'data/id_maps/{name}.csv', data/f'id_maps/{name}.csv')
    src_items = pd.read_csv(data/'id_maps/source_items.csv', dtype={'raw_id': str})
    tgt_items = pd.read_csv(data/'id_maps/target_items.csv', dtype={'raw_id': str})
    assert not set(src_items.raw_id) & set(tgt_items.raw_id), 'Cross-domain item overlap'
    for name, users in [('train', train_users), ('validation', val_users), ('test', sorted(test_users.tolist()))]:
        pd.Series(users).to_csv(data/f'{name}_users.csv', index=False, header=False)
    protocol = dict(version=1, task='Book->Movie', validation_seed=20220922,
                    test_origin='Existing seed-2020 split, preserved byte-for-byte; historically evaluated, not a new blind holdout',
                    overlap_users=n, train_overlap_users=len(train_users), validation_users=len(val_users),
                    test_users=len(test_users), target_train_users=int(train.uid.nunique()),
                    target_nonoverlap_train_users=int(train.loc[train.uid >= n, 'uid'].nunique()),
                    source_ratings=old_manifest['source_interactions'], target_train_ratings=len(train),
                    mapping_train_ratings=len(meta), validation_ratings=len(val),
                    test_ratings=old_manifest['test_interactions'], raw_item_overlap=0,
                    target_train_excludes_validation_and_test_users=True,
                    source_access='All original source ratings, including cold users source histories',
                    split='60/20/20 of overlapping users; all target-only users retained in training',
                    protocol_type='Random user holdout, full catalog ID mapping; not temporal evaluation',
                    candidate_order=NAMES, smoothed_mean_formula='(sum_i + alpha * training_global_mean)/(count_i + alpha)',
                    selection='Lowest validation MAE and RMSE separately; ties follow candidate_order',
                    final_test='Select and freeze fitted statistics before opening test labels; evaluate all predeclared candidates once',
                    refit_on_validation=False,
                    legacy_wrapper_warning='Legacy experiment.py is not compatible with this three-way split; use explicit train/validation user lists',
                    data_sha256={str(p.relative_to(root)): digest(p) for p in sorted(data.rglob('*.csv'))},
                    implementation_sha256=digest(Path(__file__)))
    assert protocol['data_sha256']['data/test.csv'] == digest(OLD/'test.csv')
    assert len(train) + len(val) + protocol['test_ratings'] == 1697533
    save(root/'protocol.json', protocol)
    print('PROTOCOL_PREPARED', json.dumps(protocol), flush=True)


def fit(train):
    grouped = train.groupby('iid').y
    return dict(ids=np.sort(train.iid.unique()), counts=grouped.size().to_numpy(),
                sums=grouped.sum().to_numpy(dtype=float), medians=grouped.median().to_numpy(),
                global_mean=np.asarray(train.y.mean()), global_median=np.asarray(train.y.median()))


def predict(model, ids, name):
    if name.startswith('global_'):
        return np.full(len(ids), float(model[name]))
    index = np.searchsorted(model['ids'], ids)
    safe = np.minimum(index, len(model['ids']) - 1)
    known = (index < len(model['ids'])) & (model['ids'][safe] == ids)
    count = np.where(known, model['counts'][safe], 0)
    sums = np.where(known, model['sums'][safe], 0.)
    mean = float(model['global_mean'])
    if name == 'item_median':
        return np.where(known, model['medians'][safe], float(model['global_median']))
    alpha = float(name.rsplit('_', 1)[1]) if name.startswith('smoothed_mean_') else 0.
    return np.divide(sums + alpha*mean, count + alpha,
                     out=np.full(len(ids), mean), where=(count + alpha) != 0)


def metric(pred, frame):
    labels = frame.y.to_numpy()
    assert pred.shape == labels.shape and np.isfinite(pred).all()
    error = pred - labels
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                user_macro_mae=float(pd.Series(np.abs(error)).groupby(frame.uid.to_numpy()).mean().mean()))


def check_hashes(root, protocol, names):
    for name in names:
        assert digest(root/name) == protocol['data_sha256'][name], f'Data changed: {name}'


def validate(root):
    protocol = json.loads((root/'protocol.json').read_text())
    assert not (root/'selection.json').exists()
    denied = {str((root/'data/test.csv').resolve()), str((OLD/'test.csv').resolve()),
              str((BASE/'data/mid/Movies_and_TV.csv').resolve()),
              str((BASE/'data/raw/reviews_Movies_and_TV_5.json.gz').resolve())}
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            if str(Path(os.fsdecode(args[0])).resolve()) in denied:
                raise RuntimeError('Test-label read forbidden during fit/validation')
    sys.addaudithook(guard)
    check_hashes(root, protocol, ['data/train_tgt.csv', 'data/validation.csv'])
    train = read(root/'data/train_tgt.csv')
    model = fit(train)
    np.savez(root/'models/baseline_statistics.npz', **model)
    val = read(root/'data/validation.csv')
    assert not set(train.uid) & set(val.uid)
    scores = {name: metric(predict(model, val.iid.to_numpy(), name), val) for name in NAMES}
    # Independent reference and unseen-item fallback checks.
    reference = val.iid.map(train.groupby('iid').y.mean()).fillna(train.y.mean()).to_numpy()
    np.testing.assert_allclose(predict(model, val.iid.to_numpy(), 'item_mean'), reference, rtol=0, atol=1e-12)
    unknown = np.array([int(train.iid.max()) + 1])
    assert predict(model, unknown, 'item_mean')[0] == train.y.mean()
    winners = {key: min(NAMES, key=lambda name: scores[name][key]) for key in ['mae', 'rmse']}
    selection = dict(validation_metrics=scores, selected_by_validation=winners,
                     test_labels_opened=False, test_access_guard_enabled=True,
                     fitted_statistics_sha256=digest(root/'models/baseline_statistics.npz'),
                     protocol_sha256=digest(root/'protocol.json'))
    save(root/'selection.json', selection)
    print('VALIDATION_SELECTION_FROZEN', json.dumps(selection), flush=True)


def test(root):
    protocol = json.loads((root/'protocol.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    assert digest(root/'protocol.json') == selection['protocol_sha256']
    assert digest(root/'models/baseline_statistics.npz') == selection['fitted_statistics_sha256']
    # A failed test evaluation must be inspected rather than silently rerun/tuned.
    with (root/'test_started.json').open('x') as stream:
        json.dump(dict(selection_sha256=digest(root/'selection.json'), job_id=os.environ.get('SLURM_JOB_ID')), stream)
    check_hashes(root, protocol, ['data/test.csv', 'data/train_users.csv', 'data/validation_users.csv', 'data/test_users.csv'])
    frame = read(root/'data/test.csv')
    expected = pd.read_csv(root/'data/test_users.csv', header=None).to_numpy().ravel()
    np.testing.assert_array_equal(np.unique(frame.uid), expected)
    assert len(frame) == protocol['test_ratings']
    model = dict(np.load(root/'models/baseline_statistics.npz'))
    predictions = {name: predict(model, frame.iid.to_numpy(), name) for name in NAMES}
    scores = {name: metric(predictions[name], frame) for name in NAMES}
    np.savez_compressed(root/'results/test_predictions.npz', uid=frame.uid.to_numpy(),
                        iid=frame.iid.to_numpy(), y=frame.y.to_numpy(), **predictions)
    result = dict(status='PASS', job_id=os.environ.get('SLURM_JOB_ID'),
                  selected_by_validation=selection['selected_by_validation'], test_metrics=scores,
                  selected_test_metrics={key: scores[name] for key, name in selection['selected_by_validation'].items()},
                  test_evaluations=1, target_test_users=len(expected), target_test_ratings=len(frame),
                  unseen_test_ratings=int((~np.isin(frame.iid, model['ids'])).sum()))
    save(root/'results/summary.json', result)
    lines = ['# Book→Movie 冷启动验证协议：第一优先级实验', '',
             '状态：PASS。此轮完成固定三路划分和目标侧基线，未训练跨域模型。', '',
             f"重叠用户训练 / 验证 / 测试：{protocol['train_overlap_users']} / {protocol['validation_users']} / {protocol['test_users']}。",
             f"目标评分训练 / 验证 / 测试：{protocol['target_train_ratings']} / {protocol['validation_ratings']} / {protocol['test_ratings']}。", '',
             'Book/Movie 原始物品 ID 交集为 0；目标训练排除全部验证和测试用户。',
             '测试沿用历史 seed2020 用户及评分文件，不宣称是从未查看过的新盲测。',
             '数据使用随机用户切分，没有时间外推约束。', '',
             '训练统计量仅由新 train_tgt 计算；验证阶段禁止读取测试评分。',
             '候选列表在运行前固定，按验证 MAE/RMSE 分别选择后保存；没有加入验证评分重拟合。', '',
             '| 基线 | 验证 MAE | 验证 RMSE | 测试 MAE | 测试 RMSE |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        v, t = selection['validation_metrics'][name], scores[name]
        lines.append(f"| {name} | {v['mae']:.6f} | {v['rmse']:.6f} | {t['mae']:.6f} | {t['rmse']:.6f} |")
    lines += ['', '验证集选择：' + json.dumps(selection['selected_by_validation']), '',
              '全量指标含按用户等权的 MAE，见 results/summary.json。',
              '后续模型必须使用本目录的 train_users / validation_users / test_users，',
              '仅以 validation.csv 选择 epoch 和超参数。旧 wrapper 只支持二路划分，不能直接用于本协议。',
              '源域全部训练评分允许包含冷用户历史；目标侧的验证/测试评分只用于对应评估。',
              '训练数据量已变化，因此旧协议的模型指标不能与本次基线直接比较。']
    (root/'report.md').write_text('\n'.join(lines) + '\n')
    print('PRIORITY_ONE_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'validate', 'test'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.root.resolve())
