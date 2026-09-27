# Recheck the archived run without modifying its hash-locked verifier.
# The only relaxed assertion is choices.mtime <= manifest.mtime: sequential
# writes may have equal filesystem timestamps. freeze() writes choices, then
# hashes their closed file into the manifest before test evaluation begins.
# Preserve all content hashes, all other chronology checks and all metrics.
"""Independently recompute split, one-factor controls, selection and metrics."""
import argparse
import os
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(pred, y):
    return dict(mae=float(np.abs(pred - y).mean()), rmse=float(np.sqrt(np.square(pred - y).mean())))


def verify(root):
    cfg = read(root / 'protocol.json')
    defaults = dict(train_mala_step=.25, eval_mala_step=.25, train_steps=20,
                    eval_steps=20, train_chains=32, eval_chains=128, energy_bound=5.)
    assert cfg['defaults'] == defaults
    assert cfg['hidden'] == cfg['hidden_width_fixed'] == 64
    assert cfg['seeds'] == [2020] and cfg['betas'] == [20, 80]
    original_path = next(Path(p) for p in cfg['parent_protocol_hashes'] if p.endswith('list_ambiguity_8913'))
    old = read(original_path / 'protocol.json')
    for key in ['dim', 'hidden', 'dropout', 'k', 'epochs', 'lr', 'weight_decay',
                'batch_size', 'lambdas', 'user_batch_size', 'latent_std', 'logvar_bounds',
                'energy_bound', 'train_chains', 'eval_chains', 'mala_step', 'eval_user_batch',
                'observation_variance']:
        assert cfg[key] == old[key], key
    for path, expected in cfg['code_hashes'].items():
        assert digest(Path(path)) == expected, path
    for path, expected in cfg['parent_protocol_hashes'].items():
        assert digest(Path(path) / 'protocol.json') == expected
    assert digest(Path(cfg['sensitivity_parent']) / 'protocol.json') == cfg['sensitivity_parent_sha256']
    assert read(root / 'selftest_cuda.json')['status'] == 'PASS'
    assert read(root / 'default_reproduction.json')['status'] == 'PASS'
    assert read(root / 'data_verification.json')['cases_sha256'] == digest(root / 'cases.json')
    cases = read(root / 'cases.json')
    assert len(cases) == 90 and [c['case_id'] for c in cases] == list(range(90))
    ready = read(root / 'evaluation_ready.json')
    stamp = (root / 'evaluation_ready.json').stat().st_mtime_ns
    assert len(ready['models']) == 90
    assert (root / 'cases.json').stat().st_mtime_ns < stamp
    assert digest(root / 'validation_choices.json') == ready['choices_sha256']
    assert (root / 'validation_choices.json').stat().st_mtime_ns <= stamp
    for group in range(7):
        frozen_path = root / f'group_{group}_frozen.json'
        frozen = read(frozen_path)
        assert set(frozen['models']) == {c['output'] for c in cases if c['group'] == group}
        assert frozen_path.stat().st_mtime_ns < stamp
        if group > 0:
            previous_stamp = (root / f'group_{group-1}_frozen.json').stat().st_mtime
            for case in cases:
                if case['group'] == group:
                    assert read(root / case['output'] / 'started.json')['time'] >= previous_stamp
    sources = read(root / 'source_records.json')
    for source in sources.values():
        assert digest(Path(source['path'])) == source['sha256']
    refs = read(root / 'references.json')
    for ref in refs.values():
        for name, expected in ref['hashes'].items():
            assert digest(Path(ref['path']) / name) == expected
    summary = read(root / 'summary.json')
    assert summary['status'] == 'PASS'
    rows = summary['rows']
    assert len(rows) == 126 and len(summary['sensitivity']) == 42
    choices = read(root / 'validation_choices.json')
    assert summary['choices'] == choices and len(choices) == 42
    audited = []
    for task, overlap in cfg['tasks'].items():
        for beta in cfg['betas']:
            cell = root / task / f'beta_{beta}'
            manifest = read(cell / 'manifest.json')
            assert manifest['cross_domain_raw_item_overlap'] == 0 and manifest['overlap_users'] == overlap
            for path, expected in manifest['hashes'].items():
                assert digest(cell / path) == expected
            inner, train, val, test = [pd.read_csv(cell / f'data/{name}.csv', header=None, names=['u', 'i', 'r'])
                                     for name in ['inner_train', 'train', 'validation', 'test']]
            assert not set(inner.u) & set(val.u)
            assert not set(train.u) & set(test.u)
            assert not set(val.u) & set(test.u)
            np.testing.assert_array_equal(test[['u', 'i']].to_numpy(), np.load(cell / 'data/test_features.npy'))
            means = test.i.map(train.groupby('i').r.mean()).fillna(train.r.mean()).to_numpy()
            vmeans = val.i.map(inner.groupby('i').r.mean()).fillna(inner.r.mean()).to_numpy()
            baseval = metric(vmeans, val.r.to_numpy())['rmse']
            default = next(c for c in cases if c['task'] == task and c['beta'] == beta and c['axis'] == 'default')
            dscores = metric(np.load(root / default['output'] / 'test.npy'), test.r.to_numpy())
            listref = Path(refs[f'{task}/{beta}/LIST_v1']['path'])
            lscores = metric(np.load(listref / 'test.npy'), test.r.to_numpy())
            commonhash = read(listref / 'complete.json')['initial_parameters_sha256']
            baseline = next(r for r in summary['baselines'] if r['task'] == task and r['beta'] == beta)
            for label, scores in [('list', lscores), ('energy', dscores), ('mean', metric(means, test.r.to_numpy()))]:
                for name, value in scores.items():
                    assert abs(baseline[label + '_' + name] - value) < 1e-12
            initialization = None
            for case in cases:
                if case['task'] != task or case['beta'] != beta:
                    continue
                changed = [key for key in defaults if case['controls'][key] != defaults[key]]
                assert changed == ([] if case['axis'] == 'default' else [case['axis']])
                if changed:
                    assert case['controls'][case['axis']] == case['value']
                    assert case['value'] in cfg['axes'][case['axis']]['values']
                    assert case['group'] == cfg['axes'][case['axis']]['group']
                assert read(root / case['branch'] / 'protocol.json') == dict(cfg, task=task, **case['controls'])
                out = root / case['output']
                done = read(out / 'complete.json')
                assert done['status'] == 'PASS' and done['test_labels_opened'] is False
                assert done['refit_reinitialized'] and done['restored_predictions']
                assert done['common_initial_parameters_sha256'] == commonhash
                initialization = initialization or done['initial_parameters_sha256']
                assert done['initial_parameters_sha256'] == initialization
                assert done['source_ids_sha256'] == sources[task]['sha256']
                assert done['full_outer_train_rows'] == len(train)
                assert done['overlap_train_rows'] == int((train.u < overlap).sum())
                assert done['grouped_users'] == manifest['users']['outer_train']
                assert digest(out / 'complete.json') == ready['models'][case['output']]
                assert (out / 'complete.json').stat().st_mtime_ns < stamp
                for name, key in [('final.pt', 'final_sha256'), ('test.npy', 'test_sha256'), ('selection.json', 'selection_sha256')]:
                    assert digest(out / name) == done[key]
                    assert (out / name).stat().st_mtime_ns < stamp
                selected = dict(epoch=0, lambda_value=0., validation_rmse=baseval)
                history = read(out / 'select_history.json')
                assert [r['epoch'] for r in history] == list(range(1, 11))
                for row in history:
                    assert set(row['validation']) == {str(x) for x in cfg['lambdas']}
                    for lam in cfg['lambdas']:
                        value = row['validation'][str(lam)]
                        if value < selected['validation_rmse']:
                            selected = dict(epoch=row['epoch'], lambda_value=lam, validation_rmse=value)
                actual = read(out / 'selection.json')
                assert selected['epoch'] == actual['epoch'] and selected['lambda_value'] == actual['lambda_value']
                assert abs(selected['validation_rmse'] - actual['validation_rmse']) < 1e-12
                assert abs(metric(np.load(out / 'validation.npy'), val.r.to_numpy())['rmse'] - actual['validation_rmse']) < 1e-12
                assert len(read(out / 'refit_history.json')) == done['refit_epochs'] == actual['epoch']
                pred = np.load(out / 'test.npy')
                assert pred.shape == (len(test),) and np.isfinite(pred).all()
                unseen = ~test.i.isin(train.i.unique()).to_numpy()
                np.testing.assert_array_equal(pred[unseen], means[unseen])
                linked = [r for r in rows if r['case_id'] == case['case_id']]
                assert len(linked) == (7 if case['axis'] == 'default' else 1)
                scores = metric(pred, test.r.to_numpy())
                for r in linked:
                    for key, value in scores.items():
                        assert abs(r[key] - value) < 1e-12
                        assert abs(r[f'delta_{key}_vs_default'] - (value - dscores[key])) < 1e-12
                        assert abs(r[f'delta_{key}_vs_LIST_v1'] - (value - lscores[key])) < 1e-12
                if case['axis'] == 'default':
                    ref = Path(refs[f'{task}/{beta}/energy_previous']['path'])
                    np.testing.assert_allclose(pred, np.load(ref / 'test.npy'), atol=1e-7, rtol=0)
                    assert actual == read(ref / 'selection.json')
                audited.append(case['case_id'])
            for key, axis in cfg['axes'].items():
                axisrows = [r for r in rows if r['task'] == task and r['beta'] == beta and r['axis'] == key]
                assert [r['value'] for r in axisrows] == axis['values']
                chosen = min(axisrows, key=lambda r: r['validation_rmse'])
                recorded = next(r for r in choices if r['task'] == task and r['beta'] == beta and r['axis'] == key)
                assert chosen['case_id'] == recorded['case_id'] and chosen['value'] == recorded['value']
                assert [r['case_id'] for r in axisrows if r['validation_selected']] == [chosen['case_id']]
                sensitivity = next(s for s in summary['sensitivity'] if s['task'] == task and s['beta'] == beta and s['axis'] == key)
                for name in ['mae', 'rmse']:
                    assert abs(sensitivity[name + '_range'] - (max(r[name] for r in axisrows) - min(r[name] for r in axisrows))) < 1e-12
                    assert abs(sensitivity['max_abs_delta_' + name] - max(abs(r[f'delta_{name}_vs_default']) for r in axisrows)) < 1e-12
    assert len(set(audited)) == 90
    result = dict(status='PASS', models=90, rows=126, seed=2020, hidden_width=64,
        checks=['six sequential groups with seven independent axes; no accumulated tuning',
                'one scalar changes per candidate; all other defaults and architecture fixed',
                'exact archived-default parity and independent train/inference controls',
                'same initialization/source lists/cold splits and training-only ordinary means',
                'validation-only epoch/lambda choices and fresh refit',
                'all candidates and validation choices frozen before test evaluation',
                'independently recomputed metrics/deltas/sensitivity and unseen-item fallback'],
        verifier_sha256=digest(Path(__file__)),
        original_verifier_sha256=digest(Path(__file__).with_name('verify_list_langevin_sensitivity.py')),
        recovery_reason='sequential choice/manifest writes share one filesystem timestamp; allow equality while retaining the frozen content hash and all model timestamp checks',
        choice_mtime_ns=(root / 'validation_choices.json').stat().st_mtime_ns,
        freeze_mtime_ns=stamp,
        review_slurm_job=os.environ.get('SLURM_JOB_ID'),
        model_and_prediction_files_changed=False)
    (root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    verify(parser.parse_args().root.resolve())
