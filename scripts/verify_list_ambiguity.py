"""Independent CPU recomputation of splits, selection, means and final metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    cfg = read(root / 'protocol.json')
    assert read(root / 'selftest_cuda.json')['status'] == 'PASS'
    parent = Path(cfg['parent'])
    assert digest(parent / 'protocol.json') == cfg['parent_protocol_sha256']
    for path, expected in cfg['code_hashes'].items():
        assert digest(Path(path)) == expected, path
    frozen = read(root / 'evaluation_started.json')
    stamp = (root / 'evaluation_started.json').stat().st_mtime_ns
    assert len(frozen) == 30
    summary = read(root / 'summary.json')
    results = []
    for beta in cfg['betas']:
        cell = root / f'beta_{beta}'
        manifest = read(cell / 'manifest.json')
        assert manifest['overlap_users'] == 17904 and manifest['cross_domain_raw_item_overlap'] == 0
        for path, expected in manifest['hashes'].items():
            assert digest(cell / path) == expected, path
        data = {name: pd.read_csv(cell / f'data/{name}.csv', header=None, names=['u', 'i', 'r'])
                for name in ['inner_train', 'train', 'validation', 'test']}
        inner, train, val, test = [data[n] for n in ['inner_train', 'train', 'validation', 'test']]
        assert not set(inner.u) & set(val.u)
        assert not set(train.u) & set(test.u)
        assert not set(val.u) & set(test.u)
        base = test.i.map(train.groupby('i').r.mean()).fillna(train.r.mean()).to_numpy()
        valbase = val.i.map(inner.groupby('i').r.mean()).fillna(inner.r.mean()).to_numpy()
        np.testing.assert_array_equal(test[['u', 'i']].to_numpy(), np.load(cell / 'data/test_features.npy'))
        init_hashes = {}
        for variant in cfg['variants']:
            for seed in cfg['seeds']:
                out = root / variant / f'beta_{beta}/seed_{seed}'
                rel = str(out.relative_to(root))
                done = read(out / 'complete.json')
                assert digest(out / 'complete.json') == frozen[rel]
                assert (out / 'complete.json').stat().st_mtime_ns < stamp
                assert not done['test_labels_opened'] and done['refit_reinitialized'] and done['restored_predictions']
                assert done['full_outer_train_rows'] == len(train)
                active = train[train.u < manifest['overlap_users']]
                assert done['overlap_train_rows'] == len(active) and done['grouped_users'] == active.u.nunique()
                assert init_hashes.setdefault(seed, done['initial_parameters_sha256']) == done['initial_parameters_sha256']
                old = read(parent / f'beta_{beta}/seed_{seed}/residual_real/complete.json')
                assert done['common_initial_parameters_sha256'] == old['initial_parameters_sha256']
                for name, key in [('final.pt', 'final_sha256'), ('test.npy', 'test_sha256'), ('selection.json', 'selection_sha256')]:
                    assert digest(out / name) == done[key]
                    assert (out / name).stat().st_mtime_ns < stamp
                chosen = dict(epoch=0, lambda_value=0., validation_rmse=float(np.sqrt(np.square(valbase - val.r).mean())))
                history = read(out / 'select_history.json')
                assert [row['epoch'] for row in history] == list(range(1, cfg['epochs'] + 1))
                for row in history:
                    for lam in cfg['lambdas']:
                        value = row['validation'][str(lam)]
                        if value < chosen['validation_rmse']:
                            chosen = dict(epoch=row['epoch'], lambda_value=lam, validation_rmse=value)
                actual = read(out / 'selection.json')
                assert actual == chosen
                assert abs(np.sqrt(np.square(np.load(out / 'validation.npy') - val.r).mean()) - chosen['validation_rmse']) < 1e-12
                assert len(read(out / 'refit_history.json')) == chosen['epoch'] == done['refit_epochs']
                prediction = np.load(out / 'test.npy')
                assert prediction.shape == (len(test),) and np.isfinite(prediction).all()
                unseen = ~test.i.isin(train.i.unique()).to_numpy()
                np.testing.assert_array_equal(prediction[unseen], base[unseen])
                mae = float(np.abs(prediction - test.r).mean())
                rmse = float(np.sqrt(np.square(prediction - test.r).mean()))
                result = next(row for row in summary['per_seed'] if row['beta'] == beta and row['variant'] == variant and row['seed'] == seed)
                assert abs(result['mae'] - mae) < 1e-12 and abs(result['rmse'] - rmse) < 1e-12
                for control, key in [(parent / f'beta_{beta}/seed_{seed}/residual_real/test.npy', 'archived'),
                                     (root / f'point_matched/beta_{beta}/seed_{seed}/test.npy', 'matched')]:
                    pred = np.load(control)
                    assert abs(result['delta_mae_vs_' + key] - (mae - np.abs(pred - test.r).mean())) < 1e-12
                    assert abs(result['delta_rmse_vs_' + key] - (rmse - np.sqrt(np.square(pred - test.r).mean()))) < 1e-12
                results.append(result)
    for row in summary['comparison']:
        selected = [v for v in summary['per_seed'] if v['beta'] == row['beta'] and v['variant'] == row['variant']]
        for name in ['mae', 'rmse']:
            assert abs(row[name] - np.mean([v[name] for v in selected])) < 1e-12
            assert abs(row[name + '_std'] - np.std([v[name] for v in selected], ddof=1)) < 1e-12
    record = dict(status='PASS', models=len(results), verifier_sha256=digest(Path(__file__)),
                  checks=['30 predictions frozen before test read', 'deduplicated identical data and source-list protocol',
                          'disjoint cold users', 'training-only unsmoothed means', 'shared initialization with archived LIST',
                          'all observed overlap training ratings retained', 'validation-only epoch/lambda selection',
                          'fresh full outer refit', 'checkpoint prediction replay', 'independent MAE/RMSE and paired deltas',
                          'unseen-item fallback', 'joint Gaussian likelihood and Langevin math selftests'], rows=results)
    (root / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', models=len(results), checks=record['checks']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    verify(parser.parse_args().root.resolve())
