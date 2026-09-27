"""Independent result audit for the two-dataset conditional LIST extension."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(root, task, variant, beta, seed):
    if variant == 'LIST_v1':
        return root / task / f'beta_{beta}/seed_{seed}/residual_real'
    return root / task / variant / f'beta_{beta}/seed_{seed}'


def metrics(pred, y):
    return dict(mae=float(np.abs(pred - y).mean()), rmse=float(np.sqrt(np.square(pred - y).mean())))


def verify(root):
    cfg = read(root / 'protocol.json')
    old_cfg = read(Path(cfg['model_parent']) / 'protocol.json')
    assert digest(Path(cfg['model_parent']) / 'protocol.json') == cfg['model_parent_protocol_sha256']
    assert digest(Path(cfg['data_parent']) / 'protocol.json') == cfg['data_parent_protocol_sha256']
    assert read(root / 'selftest_cuda.json')['status'] == 'PASS'
    for name in ['dim', 'hidden', 'dropout', 'k', 'epochs', 'lr', 'weight_decay', 'batch_size',
                 'lambdas', 'user_batch_size', 'latent_std', 'logvar_bounds', 'energy_bound',
                 'train_chains', 'eval_chains', 'mala_step', 'eval_user_batch', 'observation_variance']:
        assert cfg[name] == old_cfg[name], name
    for path, expected in cfg['code_hashes'].items():
        assert digest(Path(path)) == expected, path
    for record in read(root / 'source_records.json').values():
        assert digest(Path(record['path'])) == record['sha256']
    frozen = read(root / 'evaluation_started.json')
    stamp = (root / 'evaluation_started.json').stat().st_mtime_ns
    assert len(frozen) == 48
    summary = read(root / 'summary.json')
    audited, steps = [], []
    for task, overlap in cfg['tasks'].items():
        for beta in cfg['betas']:
            cell = root / task / f'beta_{beta}'
            manifest = read(cell / 'manifest.json')
            assert manifest['overlap_users'] == overlap and manifest['cross_domain_raw_item_overlap'] == 0
            for path, expected in manifest['hashes'].items():
                assert digest(cell / path) == expected, path
            data = {name: pd.read_csv(cell / f'data/{name}.csv', header=None, names=['u', 'i', 'r'])
                    for name in ['inner_train', 'train', 'validation', 'test']}
            inner, train, val, test = [data[name] for name in ['inner_train', 'train', 'validation', 'test']]
            assert not set(inner.u) & set(val.u)
            assert not set(train.u) & set(test.u)
            assert not set(val.u) & set(test.u)
            assert len(set(train.u) & set(range(overlap))) == manifest['users']['outer_train']
            np.testing.assert_array_equal(test[['u', 'i']].to_numpy(), np.load(cell / 'data/test_features.npy'))
            base = test.i.map(train.groupby('i').r.mean()).fillna(train.r.mean()).to_numpy()
            valbase = val.i.map(inner.groupby('i').r.mean()).fillna(inner.r.mean()).to_numpy()
            steps.append(dict(task=task, beta=beta,
                              LIST_v1_updates_per_inner_epoch=math.ceil(len(inner) / cfg['batch_size']),
                              grouped_updates_per_inner_epoch=math.ceil(manifest['users']['inner_train'] / cfg['user_batch_size'])))
            initial = {}
            for variant in cfg['variants']:
                for seed in cfg['seeds']:
                    out = output(root, task, variant, beta, seed)
                    done = read(out / 'complete.json')
                    assert done['status'] == 'PASS' and not done['test_labels_opened']
                    assert done['refit_reinitialized'] and done['restored_predictions']
                    assert done['full_outer_train_rows'] == len(train)
                    rel = str(out.relative_to(root))
                    assert digest(out / 'complete.json') == frozen[rel]
                    assert (out / 'complete.json').stat().st_mtime_ns < stamp
                    for filename, key in [('final.pt', 'final_sha256'), ('selection.json', 'selection_sha256'), ('test.npy', 'test_sha256')]:
                        assert digest(out / filename) == done[key]
                        assert (out / filename).stat().st_mtime_ns < stamp
                    native = read(output(root, task, 'LIST_v1', beta, seed) / 'complete.json')
                    if variant == 'LIST_v1':
                        np.testing.assert_array_equal(np.load(out / 'assignment.npy'), np.arange(overlap))
                        assert digest(out / 'assignment.npy') == done['assignment_sha256']
                    else:
                        assert done['common_initial_parameters_sha256'] == native['initial_parameters_sha256']
                        assert initial.setdefault(seed, done['initial_parameters_sha256']) == done['initial_parameters_sha256']
                        assert done['overlap_train_rows'] == int((train.u < overlap).sum())
                        assert done['grouped_users'] == manifest['users']['outer_train']
                        assert done['source_ids_sha256'] == read(root / 'source_records.json')[f'{task}/{seed}']['sha256']
                    history = read(out / 'select_history.json')
                    assert [v['epoch'] for v in history] == list(range(1, cfg['epochs'] + 1))
                    chosen = dict(epoch=0, lambda_value=0., validation_rmse=float(np.sqrt(np.square(valbase - val.r).mean())))
                    for row in history:
                        for lam in cfg['lambdas']:
                            value = row['validation'][str(lam)]
                            if value < chosen['validation_rmse']:
                                chosen = dict(epoch=row['epoch'], lambda_value=lam, validation_rmse=value)
                    actual = read(out / 'selection.json')
                    assert actual['epoch'] == chosen['epoch'] and actual['lambda_value'] == chosen['lambda_value']
                    assert abs(actual['validation_rmse'] - chosen['validation_rmse']) < 1e-12
                    assert abs(np.sqrt(np.square(np.load(out / 'validation.npy') - val.r).mean()) - actual['validation_rmse']) < 1e-12
                    assert len(read(out / 'refit_history.json')) == actual['epoch'] == done['refit_epochs']
                    pred = np.load(out / 'test.npy')
                    assert pred.shape == (len(test),) and np.isfinite(pred).all()
                    unseen = ~test.i.isin(train.i.unique()).to_numpy()
                    np.testing.assert_array_equal(pred[unseen], base[unseen])
                    independent = metrics(pred, test.r.to_numpy())
                    row = next(v for v in summary['per_seed'] if v['task'] == task and v['beta'] == beta and v['variant'] == variant and v['seed'] == seed)
                    for key in ['mae', 'rmse']:
                        assert abs(row[key] - independent[key]) < 1e-12
                        for reference in ['LIST_v1', 'point_matched']:
                            score = metrics(np.load(output(root, task, reference, beta, seed) / 'test.npy'), test.r.to_numpy())[key]
                            assert abs(row[f'delta_{key}_vs_{reference}'] - (independent[key] - score)) < 1e-12
                    audited.append(row)
            baseline_row = next(v for v in summary['comparison'] if v['task'] == task and v['beta'] == beta and v['variant'] == 'item_mean')
            for key, value in metrics(base, test.r.to_numpy()).items():
                assert abs(baseline_row[key] - value) < 1e-12
    for row in summary['comparison']:
        selected = [v for v in summary['per_seed'] if v['task'] == row['task'] and v['beta'] == row['beta'] and v['variant'] == row['variant']]
        assert len(selected) == 3
        for key in ['mae', 'rmse']:
            assert abs(row[key] - np.mean([v[key] for v in selected])) < 1e-12
            assert abs(row[key + '_std'] - np.std([v[key] for v in selected], ddof=1)) < 1e-12
    record = dict(status='PASS', models=len(audited), verifier_sha256=digest(Path(__file__)),
        checks=['48 models frozen before test evaluation', 'unchanged8913 architecture and hyperparameters',
                'actual archived residual LIST function rerun on each task', 'identical source Top100 IDs',
                'disjoint cold users and item catalogs', 'training-only ordinary means', 'common model initialization',
                'validation-only epoch/lambda choice', 'fresh outer refit', 'unseen-item fallback',
                'checkpoint prediction replay', 'independent MAE/RMSE and paired deltas'],
        update_counts=steps, rows=audited)
    (root / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', models=len(audited), update_counts=steps), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    verify(parser.parse_args().root.resolve())
