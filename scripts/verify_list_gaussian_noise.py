"""Independent audit of the one-factor observation-noise sweep."""
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


def metrics(pred, y):
    return dict(mae=float(np.abs(pred - y).mean()), rmse=float(np.sqrt(np.square(pred - y).mean())))


def output(root, task, q, beta):
    name = 'noise_' + format(float(q), 'g').replace('.', 'p')
    return root / task / name / f'gaussian_learned/beta_{beta}/seed_2020'


def verify(root):
    cfg = read(root / 'protocol.json')
    assert cfg['seeds'] == [2020]
    assert set(cfg['observation_variances']) == {.25, .5, 1., 2., 4.}
    parents = cfg['parent_protocol_hashes']
    for path, expected in parents.items():
        assert digest(Path(path) / 'protocol.json') == expected
    old = read(next(Path(p) for p in parents if p.endswith('list_ambiguity_8913')) / 'protocol.json')
    fixed = ['dim', 'hidden', 'dropout', 'k', 'epochs', 'lr', 'weight_decay', 'batch_size',
             'lambdas', 'user_batch_size', 'latent_std', 'logvar_bounds', 'energy_bound',
             'train_chains', 'eval_chains', 'mala_step', 'eval_user_batch']
    for key in fixed:
        assert cfg[key] == old[key], key
    for path, expected in cfg['code_hashes'].items():
        assert digest(Path(path)) == expected, path
    assert read(root / 'selftest_cuda.json')['status'] == 'PASS'
    assert read(root / 'q1_reproduction.json')['status'] == 'PASS'
    source_records = read(root / 'source_records.json')
    for ref in source_records.values():
        assert digest(Path(ref['path'])) == ref['sha256']
    refs = read(root / 'references.json')
    for ref in refs.values():
        for name, expected in ref['hashes'].items():
            assert digest(Path(ref['path']) / name) == expected
    ready_path = root / 'evaluation_ready.json'
    stamp = ready_path.stat().st_mtime_ns
    ready = read(ready_path)
    assert len(ready['models']) == 30
    assert digest(root / 'noise_selection.json') == ready['selection_sha256']
    assert (root / 'noise_selection.json').stat().st_mtime_ns <= stamp
    summary = read(root / 'summary.json')
    rows = summary['rows']
    assert len(rows) == 48
    assert summary['noise_selection'] == read(root / 'noise_selection.json')
    audited, selected_rows = [], []
    for task, overlap in cfg['tasks'].items():
        for beta in cfg['betas']:
            cell = root / task / f'beta_{beta}'
            manifest = read(cell / 'manifest.json')
            assert manifest['overlap_users'] == overlap and manifest['cross_domain_raw_item_overlap'] == 0
            for path, expected in manifest['hashes'].items():
                assert digest(cell / path) == expected, path
            inner, train, val, test = [pd.read_csv(cell / f'data/{name}.csv', header=None, names=['u', 'i', 'r'])
                                       for name in ['inner_train', 'train', 'validation', 'test']]
            assert not set(inner.u) & set(val.u)
            assert not set(train.u) & set(test.u)
            assert not set(val.u) & set(test.u)
            assert len(set(train.u) & set(range(overlap))) == manifest['users']['outer_train']
            np.testing.assert_array_equal(test[['u', 'i']].to_numpy(), np.load(cell / 'data/test_features.npy'))
            base = test.i.map(train.groupby('i').r.mean()).fillna(train.r.mean()).to_numpy()
            valbase = val.i.map(inner.groupby('i').r.mean()).fillna(inner.r.mean()).to_numpy()
            basemse = float(np.sqrt(np.square(valbase - val.r.to_numpy()).mean()))
            reference_scores = {}
            for name in ['LIST_v1', 'gaussian_previous']:
                p = Path(refs[f'{task}/{beta}/{name}']['path'])
                score = metrics(np.load(p / 'test.npy'), test.r.to_numpy())
                reference_scores[name] = score
                row = next(r for r in rows if r['task'] == task and r['beta'] == beta and r['method'] == name)
                for key, value in score.items():
                    assert abs(row[key] - value) < 1e-12
            original = read(Path(refs[f'{task}/{beta}/LIST_v1']['path']) / 'complete.json')
            initial = None
            selections = []
            for q in cfg['observation_variances']:
                out = output(root, task, q, beta)
                subcfg = read(out.parents[2] / 'protocol.json')
                assert subcfg == dict(cfg, task=task, observation_variance=q)
                done = read(out / 'complete.json')
                assert done['status'] == 'PASS' and done['test_labels_opened'] is False
                assert done['refit_reinitialized'] and done['restored_predictions']
                assert done['full_outer_train_rows'] == len(train)
                assert done['overlap_train_rows'] == int((train.u < overlap).sum())
                assert done['grouped_users'] == manifest['users']['outer_train']
                assert done['source_ids_sha256'] == source_records[task]['sha256']
                assert done['common_initial_parameters_sha256'] == original['initial_parameters_sha256']
                initial = initial or done['initial_parameters_sha256']
                assert initial == done['initial_parameters_sha256']
                assert ready['models'][str(out.relative_to(root))] == digest(out / 'complete.json')
                assert (out / 'complete.json').stat().st_mtime_ns < stamp
                for filename, key in [('final.pt', 'final_sha256'), ('test.npy', 'test_sha256'), ('selection.json', 'selection_sha256')]:
                    assert digest(out / filename) == done[key]
                    assert (out / filename).stat().st_mtime_ns < stamp
                history = read(out / 'select_history.json')
                assert [r['epoch'] for r in history] == list(range(1, 11))
                chosen = dict(epoch=0, lambda_value=0., validation_rmse=basemse)
                for r in history:
                    assert set(r['validation']) == {str(x) for x in cfg['lambdas']}
                    for lam in cfg['lambdas']:
                        score = r['validation'][str(lam)]
                        if score < chosen['validation_rmse']:
                            chosen = dict(epoch=r['epoch'], lambda_value=lam, validation_rmse=score)
                actual = read(out / 'selection.json')
                assert actual['epoch'] == chosen['epoch'] and actual['lambda_value'] == chosen['lambda_value']
                assert abs(actual['validation_rmse'] - chosen['validation_rmse']) < 1e-12
                valmetric = metrics(np.load(out / 'validation.npy'), val.r.to_numpy())['rmse']
                assert abs(valmetric - actual['validation_rmse']) < 1e-12
                assert len(read(out / 'refit_history.json')) == actual['epoch'] == done['refit_epochs']
                selections.append(dict(task=task, beta=beta, noise=q, **actual))
                pred = np.load(out / 'test.npy')
                assert pred.shape == (len(test),) and np.isfinite(pred).all()
                unseen = ~test.i.isin(train.i.unique()).to_numpy()
                np.testing.assert_array_equal(pred[unseen], base[unseen])
                row = next(r for r in rows if r['task'] == task and r['beta'] == beta and r['method'] == 'gaussian_noise' and r['observation_variance'] == q)
                for key, value in metrics(pred, test.r.to_numpy()).items():
                    assert abs(row[key] - value) < 1e-12
                    for ref, score in reference_scores.items():
                        assert abs(row[f'delta_{key}_vs_{ref}'] - (value - score[key])) < 1e-12
                if q == 1.:
                    ref = Path(refs[f'{task}/{beta}/gaussian_previous']['path'])
                    np.testing.assert_allclose(pred, np.load(ref / 'test.npy'), atol=1e-7, rtol=0)
                    assert actual == read(ref / 'selection.json')
                audited.append(row)
            best = min(selections, key=lambda r: r['validation_rmse'])
            recorded = next(r for r in summary['noise_selection'] if r['task'] == task and r['beta'] == beta)
            assert best == recorded
            marked = [r for r in rows if r['task'] == task and r['beta'] == beta and r['validation_selected']]
            assert len(marked) == 1 and marked[0]['observation_variance'] == best['noise']
            selected_rows.extend(marked)
            meanrow = next(r for r in rows if r['task'] == task and r['beta'] == beta and r['method'] == 'item_mean')
            for key, value in metrics(base, test.r.to_numpy()).items():
                assert abs(meanrow[key] - value) < 1e-12
    assert len(audited) == 30 and len(selected_rows) == 6
    save = dict(status='PASS', models=30, seed=2020, verifier_sha256=digest(Path(__file__)),
                checks=['only observation variance differs', 'old model and training files unchanged',
                        'same initialization, data, source lists and training-only means',
                        'exact Gaussian likelihood values and gradients independently tested',
                        'default variance reproduces old predictions and selection',
                        'validation-only epoch/lambda/noise selection', 'fresh outer refit',
                        'all models and noise choices frozen before test evaluation',
                        'independent metrics, paired deltas, cold split and unseen-item fallback'],
                selected=selected_rows)
    (root / 'verification.json').write_text(json.dumps(save, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', models=30, selected=selected_rows), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    verify(parser.parse_args().root.resolve())
