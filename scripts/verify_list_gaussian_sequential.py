"""Audit the ordered search, frozen validation choices and independent metrics."""
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


def metrics(pred, y):
    return dict(mae=float(np.abs(pred - y).mean()), rmse=float(np.sqrt(np.square(pred - y).mean())))


def verify(root):
    cfg = read(root / 'protocol.json')
    parent = Path(cfg['sequential_parent'])
    assert digest(parent / 'protocol.json') == cfg['sequential_parent_protocol_sha256']
    old = read(parent / 'protocol.json')
    assert cfg['observation_variance'] == 1. and cfg['seeds'] == [2020]
    fixed = ['dim', 'hidden', 'dropout', 'k', 'epochs', 'lr', 'weight_decay', 'batch_size',
             'lambdas', 'user_batch_size', 'energy_bound', 'train_chains', 'eval_chains', 'mala_step', 'eval_user_batch']
    for key in fixed:
        assert cfg[key] == old[key], key
    for path, expected in cfg['code_hashes'].items():
        assert digest(Path(path)) == expected, path
    assert read(root / 'selftest_cuda.json')['status'] == 'PASS'
    defaults = dict(latent_std=.3, std_lower=.05, std_upper=2.)
    grids = {1: [.3, .1, .2, .5, 1.], 2: [.05, .01, .03, .1], 3: [2., .5, 1., 3.]}
    parameters = {1: 'latent_std', 2: 'std_lower', 3: 'std_upper'}
    assert cfg['defaults'] == defaults
    assert cfg['stage_grids'] == {str(k): v for k, v in grids.items()}
    assert cfg['stage_parameters'] == {str(k): v for k, v in parameters.items()}
    refs = read(root / 'references.json')
    for ref in refs.values():
        for name, expected in ref['hashes'].items():
            assert digest(Path(ref['path']) / name) == expected
    sources = read(root / 'source_records.json')
    for ref in sources.values():
        assert digest(Path(ref['path'])) == ref['sha256']
    ready = read(root / 'evaluation_ready.json')
    stamp = (root / 'evaluation_ready.json').stat().st_mtime_ns
    summary = read(root / 'summary.json')
    all_cases, selected, previous = [], [], {}
    for stage_id in [1, 2, 3]:
        stage = root / f'stage_{stage_id}'
        manifest = read(stage / 'manifest.json')
        cases = read(stage / 'cases.json')
        frozen = read(stage / 'frozen.json')
        assert digest(stage / 'frozen.json') == ready[str(stage_id)]
        assert digest(stage / 'selection.json') == frozen['selection_sha256']
        assert digest(stage / 'cases.json') == manifest['cases_sha256']
        assert manifest['grid'] == grids[stage_id]
        assert manifest['parameter'] == parameters[stage_id]
        if stage_id > 1:
            previous_frozen = root / f'stage_{stage_id - 1}/frozen.json'
            assert manifest['prerequisite_frozen_sha256'] == digest(previous_frozen)
            assert previous_frozen.stat().st_mtime_ns < (stage / 'manifest.json').stat().st_mtime_ns
        choices = read(stage / 'selection.json')
        assert len(choices) == 6
        for task in cfg['tasks']:
            for beta in cfg['betas']:
                group = [c for c in cases if c['task'] == task and c['beta'] == beta]
                base = previous[(task, beta)]['parameters'] if stage_id > 1 else defaults
                expected, skipped = [], []
                for value in grids[stage_id]:
                    triple = dict(base, **{parameters[stage_id]: value})
                    if 0 < triple['std_lower'] <= triple['latent_std'] <= triple['std_upper']:
                        expected.append((value, triple))
                    else:
                        skipped.append(value)
                assert [(c['value'], c['parameters']) for c in group] == expected
                assert [r['value'] for r in manifest['skipped'] if r['task'] == task and r['beta'] == beta] == skipped
                for c in group:
                    assert c['parameter'] == parameters[stage_id]
                    p = c['parameters']
                    branch_cfg = read(root / c['branch'] / 'protocol.json')
                    assert branch_cfg == dict(cfg, task=task, round=stage_id, **p,
                        logvar_bounds=[math.log(p['std_lower'] ** 2), math.log(p['std_upper'] ** 2)])
                    assert digest(root / c['branch'] / 'protocol.json') == frozen['protocols'][c['branch']]
                    out = root / c['output']
                    done = read(out / 'complete.json')
                    assert done['status'] == 'PASS' and done['test_labels_opened'] is False
                    assert done['refit_reinitialized'] and done['restored_predictions']
                    assert digest(out / 'complete.json') == frozen['models'][c['output']]
                    assert (out / 'complete.json').stat().st_mtime_ns < (stage / 'frozen.json').stat().st_mtime_ns <= stamp
                    for name, key in [('final.pt', 'final_sha256'), ('test.npy', 'test_sha256'), ('selection.json', 'selection_sha256')]:
                        assert digest(out / name) == done[key]
                        assert (out / name).stat().st_mtime_ns < stamp
                    native = read(Path(refs[f'{task}/{beta}/LIST_v1']['path']) / 'complete.json')
                    assert done['common_initial_parameters_sha256'] == native['initial_parameters_sha256']
                    assert done['source_ids_sha256'] == sources[task]['sha256']
                    if stage_id > 1:
                        incumbent = previous[(task, beta)]
                        assert c['previous_output'] == incumbent['output']
                        if p == base:
                            assert c['reused_from'] == incumbent['output']
                            assert out.resolve() == (root / incumbent['output']).resolve()
                        else:
                            assert c['reused_from'] is None and not out.is_symlink()
                    else:
                        assert c['previous_output'] is None and c['reused_from'] is None
                best = min([dict(c, **read(root / c['output'] / 'selection.json')) for c in group], key=lambda c: c['validation_rmse'])
                chosen = next(c for c in choices if c['task'] == task and c['beta'] == beta)
                assert chosen == best
                if stage_id > 1:
                    assert chosen['validation_rmse'] <= previous[(task, beta)]['validation_rmse']
        previous = {(c['task'], c['beta']): c for c in choices}
        selected.extend(choices)
        all_cases.extend(cases)
        assert manifest['new_fits'] == sum(c['reused_from'] is None for c in cases)
        assert manifest['evaluated_configurations'] == len(cases)
    audited = []
    for task, overlap in cfg['tasks'].items():
        for beta in cfg['betas']:
            cell = root / task / f'beta_{beta}'
            manifest = read(cell / 'manifest.json')
            assert manifest['overlap_users'] == overlap and manifest['cross_domain_raw_item_overlap'] == 0
            for path, expected in manifest['hashes'].items():
                assert digest(cell / path) == expected
            inner, train, val, test = [pd.read_csv(cell / f'data/{n}.csv', header=None, names=['u', 'i', 'r'])
                                       for n in ['inner_train', 'train', 'validation', 'test']]
            assert not set(inner.u) & set(val.u)
            assert not set(train.u) & set(test.u)
            assert not set(val.u) & set(test.u)
            np.testing.assert_array_equal(test[['u', 'i']].to_numpy(), np.load(cell / 'data/test_features.npy'))
            base = test.i.map(train.groupby('i').r.mean()).fillna(train.r.mean()).to_numpy()
            valbase = val.i.map(inner.groupby('i').r.mean()).fillna(inner.r.mean()).to_numpy()
            y = test.r.to_numpy()
            ref_scores = {name: metrics(np.load(Path(refs[f'{task}/{beta}/{name}']['path']) / 'test.npy'), y)
                          for name in ['LIST_v1', 'gaussian_previous']}
            ref_scores['item_mean'] = metrics(base, y)
            for name, score in ref_scores.items():
                row = next(r for r in summary['baselines'] if r['task'] == task and r['beta'] == beta and r['method'] == name)
                for key, value in score.items():
                    assert abs(row[key] - value) < 1e-12
            for c in [c for c in all_cases if c['task'] == task and c['beta'] == beta]:
                out = root / c['output']
                done = read(out / 'complete.json')
                assert done['full_outer_train_rows'] == len(train)
                assert done['overlap_train_rows'] == int((train.u < overlap).sum())
                assert done['grouped_users'] == manifest['users']['outer_train']
                history = read(out / 'select_history.json')
                assert [r['epoch'] for r in history] == list(range(1, 11))
                best = dict(epoch=0, lambda_value=0., validation_rmse=metrics(valbase, val.r.to_numpy())['rmse'])
                for r in history:
                    assert set(r['validation']) == {str(x) for x in cfg['lambdas']}
                    for lam in cfg['lambdas']:
                        value = r['validation'][str(lam)]
                        if value < best['validation_rmse']:
                            best = dict(epoch=r['epoch'], lambda_value=lam, validation_rmse=value)
                actual = read(out / 'selection.json')
                assert actual['epoch'] == best['epoch'] and actual['lambda_value'] == best['lambda_value']
                assert abs(actual['validation_rmse'] - best['validation_rmse']) < 1e-12
                assert abs(metrics(np.load(out / 'validation.npy'), val.r.to_numpy())['rmse'] - best['validation_rmse']) < 1e-12
                assert len(read(out / 'refit_history.json')) == actual['epoch'] == done['refit_epochs']
                pred = np.load(out / 'test.npy')
                unseen = ~test.i.isin(train.i.unique()).to_numpy()
                np.testing.assert_array_equal(pred[unseen], base[unseen])
                row = next(r for r in summary['rows'] if r['output'] == c['output'])
                assert row['validation_selected'] == any(s['output'] == c['output'] for s in selected)
                assert row['reused'] == (c['reused_from'] is not None)
                for key in ['latent_std', 'std_lower', 'std_upper']:
                    assert row[key] == c['parameters'][key]
                prior = metrics(np.load(root / c['previous_output'] / 'test.npy'), y) if c['previous_output'] else ref_scores['gaussian_previous']
                for key, value in metrics(pred, y).items():
                    assert abs(row[key] - value) < 1e-12
                    for ref in ['LIST_v1', 'gaussian_previous']:
                        assert abs(row[f'delta_{key}_vs_{ref}'] - (value - ref_scores[ref][key])) < 1e-12
                    assert abs(row[f'delta_{key}_vs_previous_stage'] - (value - prior[key])) < 1e-12
                if c['round'] == 1 and c['parameters'] == defaults:
                    old_path = Path(refs[f'{task}/{beta}/gaussian_previous']['path'])
                    np.testing.assert_array_equal(pred, np.load(old_path / 'test.npy'))
                    assert actual == read(old_path / 'selection.json')
                audited.append(row)
    assert len(audited) == summary['evaluated_configurations']
    assert sum(c['reused_from'] is None for c in all_cases) == summary['new_fits']
    assert len(summary['selected']) == 18
    assert summary['selected'] == [r for r in summary['rows'] if r['validation_selected']]
    record = dict(status='PASS', evaluated_configurations=len(audited), new_fits=summary['new_fits'],
        verifier_sha256=digest(Path(__file__)),
        checks=['sequential coordinate search, not Cartesian grid', 'prior stage frozen before preparing next stage',
                'only one coordinate changes within each stage', 'q=1 and unchanged training/lambda candidates',
                'invalid upper candidates skipped, no initial std clipping', 'identical configurations reused exactly',
                'same source lists/shared initialization/training-only means', 'validation-only choices and fresh refits',
                'default Gaussian reproduced exactly', 'all stages frozen before test evaluation',
                'independent cold split, metrics and paired deltas'],
        final=[r for r in summary['selected'] if r['round'] == 3])
    (root / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    verify(parser.parse_args().root.resolve())
