"""CPU-only audit of the actual mean-baseline code and reconstructed data splits.

Does not load models or modify historical results. Submit through Slurm.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CASES = [('Books', 'Movies_and_TV', 2020),
         ('Movies_and_TV', 'Books', 2020),
         ('Movies_and_TV', 'CDs_and_Vinyl', 2020),
         ('Movies_and_TV', 'CDs_and_Vinyl', 2021)]


def baseline_code():
    path = BASE / 'scripts/audit_list_pipeline.py'
    source = path.read_text()
    audit = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
    start = next(i for i, n in enumerate(audit.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'mean')
    end = next(i for i, n in enumerate(audit.body)
               if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "metrics['target_train_item_mean']")
    nodes = audit.body[start:end + 1]
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), source, nodes[0].lineno, nodes[-1].end_lineno


CODE, SOURCE, FIRST, LAST = baseline_code()


def execute_baseline(train, test):
    predictions = []
    def metric(pred, y):
        pred = np.broadcast_to(np.asarray(pred), y.shape).copy()
        assert pred.shape == y.shape and np.isfinite(pred).all()
        predictions.append(pred)
        err = pred - y
        return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt(np.square(err).mean())))
    namespace = dict(np=np, train=train, test=test, metric=metric, metrics={})
    exec(CODE, namespace)
    return namespace['metrics'], predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    src_name, tgt_name, seed = CASES[args.case]
    assert os.environ.get('PYTHONHASHSEED') == str(seed)
    target = pd.read_csv(BASE / f'data/mid/{tgt_name}.csv')
    src_u = pd.read_csv(BASE / f'data/mid/{src_name}.csv', usecols=['uid'])
    source_users, target_users = set(src_u.uid), set(target.uid)
    co = source_users & target_users
    uids = list(co) + list(target_users - co)
    items = list(set(target.iid))
    mapped = pd.DataFrame(dict(uid=target.uid.map(dict(zip(uids, range(len(uids))))),
                               iid=target.iid.map(dict(zip(items, range(len(items))))), y=target.y))
    random.seed(seed)
    cold = random.sample(list(set(range(len(co)))), round(.2 * len(co)))
    mask = mapped.uid.isin(cold).to_numpy()
    train = mapped.loc[~mask].to_numpy(dtype=np.int64)
    test = mapped.loc[mask].to_numpy(dtype=np.int64)
    assert not np.intersect1d(train[:, 0], test[:, 0]).size
    assert not set(target.loc[mask, 'uid']) & set(target.loc[~mask, 'uid'])
    assert len(train) + len(test) == len(target)
    np.testing.assert_array_equal(np.sort(np.unique(test[:, 0])), np.sort(cold))
    # The current Book->Movie split is present on disk: compare every actual row.
    disk_verified = False
    if args.case == 0:
        data = BASE / f'data/ready/_8_2/tgt_{tgt_name}_src_{src_name}'
        np.testing.assert_array_equal(train, pd.read_csv(data/'train_tgt.csv', header=None).to_numpy())
        np.testing.assert_array_equal(test, pd.read_csv(data/'test.csv', header=None).to_numpy())
        meta = pd.read_csv(data/'train_meta.csv', header=None).to_numpy()
        np.testing.assert_array_equal(meta, train[train[:, 0] < len(co)])
        for name, values in [('target_users', uids), ('target_items', items)]:
            frame = pd.read_csv(BASE / f'data/id_maps/{name}.csv', dtype={'raw_id': str})
            assert frame.raw_id.is_unique and frame.id.is_unique
            np.testing.assert_array_equal(frame.raw_id.to_numpy(), values)
            np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(len(values)))
        disk_verified = True
    metrics, predictions = execute_baseline(train, test)
    # Independently implement fit with pandas, using only the target TRAIN rows.
    fit = pd.DataFrame(train, columns=['uid', 'iid', 'y'])
    by_item = fit.groupby('iid').y.mean()
    independent = pd.Series(test[:, 1]).map(by_item).fillna(fit.y.mean()).to_numpy()
    np.testing.assert_allclose(predictions[-1], independent, rtol=0, atol=1e-12)
    changed = test.copy()
    changed[:, 2] = np.random.default_rng(991).permutation(test[:, 2])
    _, shuffled_predictions = execute_baseline(train, changed)
    for a, b in zip(predictions, shuffled_predictions):
        np.testing.assert_array_equal(a, b)
    changed[:, 2] = 1
    _, constant_predictions = execute_baseline(train, changed)
    for a, b in zip(predictions, constant_predictions):
        np.testing.assert_array_equal(a, b)
    changed_train = train.copy()
    changed_train[:, 2] = 1
    _, changed_train_predictions = execute_baseline(changed_train, test)
    assert not np.array_equal(predictions[-1], changed_train_predictions[-1])
    # Raw-ID check of cross-domain repeated (user,item,rating) records.
    source = pd.read_csv(BASE / f'data/mid/{src_name}.csv')
    raw_test = target.loc[mask]
    source = source[source.uid.isin(raw_test.uid.unique())]
    assert not source.duplicated(['uid', 'iid']).any()
    joined = raw_test.merge(source, on=['uid', 'iid'], how='left', suffixes=('', '_source'))
    cross = joined.y_source.notna()
    historical = json.loads((BASE/'results/list_audit/checks.json').read_text())[args.case]
    old = historical['metrics']['target_train_item_mean']
    counts = fit.groupby('iid').size().reindex(test[:, 1], fill_value=0).to_numpy()
    # Probe a real robustness defect: test-only IDs above the training max.
    try:
        execute_baseline(np.array([[0, 0, 4], [1, 0, 2]]), np.array([[2, 1, 5]]))
        unseen_above_max = 'handled'
    except IndexError:
        unseen_above_max = 'IndexError (robustness issue, not leakage)'
    result = dict(task=f'{src_name}->{tgt_name}', seed=seed,
                  scope='actual on-disk split plus independent reconstruction' if disk_verified else
                        'split reconstructed from downloaded mid data; historical per-sample artifacts unavailable',
                  original_baseline_lines=[FIRST, LAST],
                  original_baseline_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
                  train_ratings=len(train), test_ratings=len(test), cold_users=len(cold),
                  train_test_user_overlap=0, target_raw_user_overlap=0,
                  actual_disk_split_and_id_maps_verified=disk_verified,
                  independent_pandas_predictions_match=True,
                  shuffled_test_labels_predictions_identical=True,
                  constant_test_labels_predictions_identical=True,
                  changed_train_labels_changes_predictions=True,
                  test_item_missing_train_ratings=int((counts == 0).sum()),
                  test_item_training_count_quantiles=np.quantile(counts, [0, .25, .5, .75, 1]).tolist(),
                  metrics=metrics, historical_item_mean_metrics=old,
                  historical_metric_match=all(abs(metrics['target_train_item_mean'][k]-old[k]) < 1e-12 for k in old),
                  train_mean=float(train[:, 2].mean()), test_rating5_fraction=float((test[:, 2] == 5).mean()),
                  source_train_target_test_same_raw_user_item=int(cross.sum()),
                  source_train_target_test_same_raw_user_item_rating=int((cross & (joined.y == joined.y_source)).sum()),
                  unseen_item_above_train_max_probe=unseen_above_max)
    out = BASE/'results/mean_leakage_audit'
    out.mkdir(exist_ok=True)
    (out/f'case_{args.case}.json').write_text(json.dumps(result, indent=2)+'\n')
    print('MEAN_AUDIT_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
