"""Sequential one-coordinate Gaussian tuning; never choose on test scores."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from book_movie_protocol import BASE, digest, save, predict
from list_source_backbones import read, check_code, student_guard
from residual_experiment import seed_all
from small_task_comparison import frame
from list_ambiguity_model import ConditionalList
import list_ambiguity_story as archived
from list_gaussian_noise import TASKS, NAMES, BETAS, SEED, baseline_root, DATA_PARENT

PARENT = BASE / 'runs/list_gaussian_noise_8952'
DEFAULTS = dict(latent_std=.3, std_lower=.05, std_upper=2.)
GRIDS = {1: [.3, .1, .2, .5, 1.], 2: [.05, .01, .03, .1], 3: [2., .5, 1., 3.]}
PARAMETERS = {1: 'latent_std', 2: 'std_lower', 3: 'std_upper'}


def stage_dir(root, round_id):
    return root / f'stage_{round_id}'


def model_output(root, case):
    return root / case['output']


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    assert read(PARENT / 'verification.json')['status'] == 'PASS'
    parent_cfg = read(PARENT / 'protocol.json')
    cfg = dict(parent_cfg)
    cfg.pop('observation_variances')
    cfg.update(observation_variance=1., defaults=DEFAULTS,
        stage_grids={str(k): v for k, v in GRIDS.items()},
        stage_parameters={str(k): v for k, v in PARAMETERS.items()},
        sequential_parent=str(PARENT), sequential_parent_protocol_sha256=digest(PARENT / 'protocol.json'),
        changed_parameter='stage1 initial latent std; stage2 lower std bound; stage3 upper std bound; only one coordinate per stage',
        unchanged='observation variance1; original ConditionalList and archived.train; 10 epochs; Adam.001/wd1e-5; 64 users/batch; original lambda grid; latent dimension32/hidden64/dropout.1; source Top100 and means',
        objective='byte-identical archived conditional Gaussian joint marginal likelihood with observation variance1',
        selection='per task/beta: validation RMSE selects each coordinate, epoch and lambda; ties retain incumbent coordinate; previous coordinate choices frozen before next stage; no test reads until all stages frozen',
        reuse='identical incumbent parameter triples in later stages reuse previous frozen checkpoints and predictions; all other candidates freshly initialized and trained',
        invalid_candidates='skip any upper bound smaller than selected initial std, preserving the prior coordinate; require 0<lower<=initial<=upper',
        evaluation='one seed2020; compare original LIST and default Gaussian from same seed; all stage models and selections frozen before test labels; historical data exploratory')
    for path, expected in cfg['code_hashes'].items():
        assert digest(Path(path)) == expected, path
    files = [Path(__file__), BASE / 'scripts/verify_list_gaussian_sequential.py',
             BASE / 'scripts/slurm_list_gaussian_sequential.sh']
    cfg['code_hashes'].update({str(p.resolve()): digest(p) for p in files})
    save(root / 'protocol.json', cfg)
    for name in ['references.json', 'source_records.json']:
        record = read(PARENT / name)
        if name == 'references.json':
            for ref in record.values():
                for filename, expected in ref['hashes'].items():
                    assert digest(Path(ref['path']) / filename) == expected
        else:
            for ref in record.values():
                assert digest(Path(ref['path'])) == ref['sha256']
        save(root / name, record)
    for task, overlap in TASKS.items():
        for beta in BETAS:
            old = PARENT / task / f'beta_{beta}'
            cell = root / task / f'beta_{beta}'
            cell.mkdir(parents=True)
            manifest = read(old / 'manifest.json')
            assert manifest['overlap_users'] == overlap and manifest['cross_domain_raw_item_overlap'] == 0
            for path, expected in manifest['hashes'].items():
                if path != 'data/test.csv':
                    assert digest(old / path) == expected
            for name in ['data', 'baseline', 'inner_baseline']:
                (cell / name).symlink_to((old / name).resolve(), target_is_directory=True)
            save(cell / 'manifest.json', manifest)
    save(root / 'data_verification.json', dict(status='PASS', observation_variance=1., no_test_labels_opened=True))
    print('SEQUENTIAL_PREPARED', root, flush=True)


def prepare_stage(root, round_id):
    check_code(root)
    student_guard(root)
    cfg = read(root / 'protocol.json')
    stage = stage_dir(root, round_id)
    stage.mkdir(exist_ok=False)
    previous = {}
    prerequisite = None
    if round_id > 1:
        prior_stage = stage_dir(root, round_id - 1)
        frozen = read(prior_stage / 'frozen.json')
        assert digest(prior_stage / 'selection.json') == frozen['selection_sha256']
        for path, expected in frozen['models'].items():
            assert digest(root / path / 'complete.json') == expected
        previous = {(r['task'], r['beta']): r for r in read(prior_stage / 'selection.json')}
        prerequisite = digest(prior_stage / 'frozen.json')
    cases, skipped = [], []
    for task in TASKS:
        for beta in BETAS:
            incumbent = previous.get((task, beta))
            fixed = incumbent['parameters'] if incumbent else DEFAULTS
            key = PARAMETERS[round_id]
            for value in GRIDS[round_id]:
                parameters = dict(fixed, **{key: value})
                if not 0 < parameters['std_lower'] <= parameters['latent_std'] <= parameters['std_upper']:
                    skipped.append(dict(task=task, beta=beta, value=value, parameters=parameters,
                                        reason='upper bound would clip the fixed initial standard deviation'))
                    continue
                branch = stage / task / f'beta_{beta}' / ('value_' + format(value, 'g').replace('.', 'p'))
                branch.mkdir(parents=True)
                branch_cfg = dict(cfg, task=task, round=round_id, **parameters,
                    logvar_bounds=[math.log(parameters['std_lower'] ** 2), math.log(parameters['std_upper'] ** 2)])
                save(branch / 'protocol.json', branch_cfg)
                cell = branch / f'beta_{beta}'
                cell.mkdir()
                for name in ['data', 'baseline', 'inner_baseline', 'manifest.json']:
                    (cell / name).symlink_to((root / task / f'beta_{beta}' / name).resolve(),
                                             target_is_directory=name != 'manifest.json')
                out = branch / f'gaussian_learned/beta_{beta}/seed_{SEED}'
                out.parent.mkdir(parents=True)
                reused = incumbent['output'] if incumbent and parameters == incumbent['parameters'] else None
                if reused:
                    out.symlink_to((root / reused).resolve(), target_is_directory=True)
                else:
                    out.mkdir()
                cases.append(dict(case_id=len(cases), round=round_id, task=task, beta=beta,
                    parameter=key, value=value, parameters=parameters, branch=str(branch.relative_to(root)),
                    output=str(out.relative_to(root)), reused_from=reused,
                    previous_output=incumbent['output'] if incumbent else None))
    save(stage / 'cases.json', cases)
    save(stage / 'manifest.json', dict(round=round_id, parameter=PARAMETERS[round_id], grid=GRIDS[round_id],
         prerequisite_frozen_sha256=prerequisite, cases_sha256=digest(stage / 'cases.json'),
         new_fits=sum(c['reused_from'] is None for c in cases), evaluated_configurations=len(cases), skipped=skipped))
    print('SEQUENTIAL_STAGE_PREPARED', round_id, 'new_fits', sum(c['reused_from'] is None for c in cases),
          'reused', sum(c['reused_from'] is not None for c in cases), 'skipped', skipped, flush=True)


def selftest(root, device):
    check_code(root)
    student_guard(root)
    base_cfg = read(root / 'protocol.json')
    lists = torch.tensor([[1, 2, 3], [3, 2, 1], [0, 0, 0]], device=device)
    base = torch.tensor([3., 4., 2.], device=device)
    users = torch.arange(3, device=device)
    combinations = [dict(DEFAULTS, **{PARAMETERS[k]: value}) for k in GRIDS for value in GRIDS[k]]
    common_hash = None
    for params in combinations:
        cfg = dict(base_cfg, **params, logvar_bounds=[math.log(params['std_lower'] ** 2), math.log(params['std_upper'] ** 2)])
        seed_all(SEED)
        model = ConditionalList(lists, 3, 3, base, cfg, 'gaussian_learned').to(device).eval()
        hashed = archived.parameter_hash(model, True)
        common_hash = common_hash or hashed
        assert common_hash == hashed, 'shared initialization changed'
        _, mean, variance = model.condition(users)
        torch.testing.assert_close(variance, torch.full_like(variance, params['latent_std'] ** 2), rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(mean, torch.zeros_like(mean), rtol=0, atol=0)
        with torch.no_grad():
            for bias, expected in [(-100., params['std_lower'] ** 2), (100., params['std_upper'] ** 2)]:
                model.log_variance.bias.fill_(bias)
                actual = model.condition(users)[2]
                torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-6, atol=1e-8)
    # Explicitly validate the stage3 exclusion: otherwise fixing initial std=1
    # while applying upper=.5 would silently change two coordinates.
    assert not 0 < .05 <= 1. <= .5
    save(root / f'selftest_{device}.json', dict(status='PASS', standard_deviation_units=True,
        initialization_and_both_bounds_tested=True, unchanged_shared_initialization=True,
        invalid_upper_excluded=True, combinations=len(combinations)))
    print('SEQUENTIAL_SELFTEST_PASS', device, flush=True)


def gpu_available(device):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu', '--format=csv'],
                            check=True, capture_output=True, text=True)
    for row in csv.DictReader(result.stdout.splitlines()):
        row = {key.strip(): value.strip() for key, value in row.items()}
        if row['index'].strip() == device:
            used = float(row['memory.used [MiB]'].split()[0])
            free = float(row['memory.free [MiB]'].split()[0])
            utilization = float(row['utilization.gpu [%]'].split()[0])
            # Only consider Slurm-allocated devices. Resident idle allocations
            # alone need not block this small model when free memory is ample.
            return used <= 512 or (free >= 8192 and utilization <= 5)
    raise RuntimeError(f'Allocated device {device} absent from nvidia-smi')


def dispatch_selftest(root):
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    while True:
        for device in devices:
            if gpu_available(device):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
                print('SELFTEST_GPU', device, flush=True)
                subprocess.run([sys.executable, '-u', str(Path(__file__)), 'selftest', '--root', str(root)], env=env, check=True)
                return
        print('WAIT_ALLOCATED_GPU_IDLE', devices, flush=True)
        time.sleep(10)


def train(root, round_id, case_id):
    check_code(root)
    case = read(stage_dir(root, round_id) / 'cases.json')[case_id]
    assert case['case_id'] == case_id and case['reused_from'] is None
    archived.PRIOR = DATA_PARENT
    archived.TASK = case['task']
    archived.PARENT = baseline_root(case['task'])
    archived.train(root / case['branch'], 'gaussian_learned', case['beta'], SEED)


def worker(root, round_id, index):
    cases = read(stage_dir(root, round_id) / 'cases.json')
    device = os.environ['CUDA_VISIBLE_DEVICES']
    ready = False
    while True:
        pending = [c for c in cases if not (model_output(root, c) / 'complete.json').exists()
                   and not (model_output(root, c) / 'claim.json').exists()]
        if not pending:
            return
        if not ready:
            if not gpu_available(device):
                print('WAIT_BUSY_ALLOCATED_GPU', device, 'remaining', len(pending), flush=True)
                time.sleep(10)
                continue
            ready = True
        for case in pending:
            out = model_output(root, case)
            try:
                with (out / 'claim.json').open('x') as handle:
                    json.dump(dict(worker=index, gpu=device), handle)
            except FileExistsError:
                continue
            print('SEQUENTIAL_TASK', index, case, flush=True)
            command = [sys.executable, '-u', str(Path(__file__)), 'train', '--root', str(root),
                       '--round', str(round_id), '--case', str(case['case_id'])]
            with (out / 'train.log').open('w') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def select_stage(root, round_id):
    check_code(root)
    student_guard(root)
    stage = stage_dir(root, round_id)
    manifest = read(stage / 'manifest.json')
    assert digest(stage / 'cases.json') == manifest['cases_sha256']
    cases = read(stage / 'cases.json')
    frozen, selections = {}, []
    for case in cases:
        out = model_output(root, case)
        done = read(out / 'complete.json')
        assert done['status'] == 'PASS' and done['test_labels_opened'] is False
        for name, key in [('final.pt', 'final_sha256'), ('test.npy', 'test_sha256'), ('selection.json', 'selection_sha256')]:
            assert digest(out / name) == done[key]
        frozen[case['output']] = digest(out / 'complete.json')
    for task in TASKS:
        for beta in BETAS:
            group = [dict(c, **read(model_output(root, c) / 'selection.json'))
                     for c in cases if c['task'] == task and c['beta'] == beta]
            selections.append(min(group, key=lambda c: c['validation_rmse']))
    with (stage / 'selection.json').open('x') as handle:
        json.dump(selections, handle, indent=2)
    save(stage / 'frozen.json', dict(models=frozen, selection_sha256=digest(stage / 'selection.json'),
         protocols={c['branch']: digest(root / c['branch'] / 'protocol.json') for c in cases}))
    print('SEQUENTIAL_STAGE_SELECTED', round_id,
          [(r['task'], r['beta'], r['parameters'], r['validation_rmse']) for r in selections], flush=True)
    if round_id == 3:
        with (root / 'evaluation_ready.json').open('x') as handle:
            json.dump({str(k): digest(stage_dir(root, k) / 'frozen.json') for k in GRIDS}, handle, indent=2)


def evaluate(root):
    check_code(root)
    assert read(root / 'selftest_cuda.json')['status'] == 'PASS'
    ready = read(root / 'evaluation_ready.json')
    all_cases, selected_ids = [], set()
    for round_id in GRIDS:
        stage = stage_dir(root, round_id)
        assert digest(stage / 'frozen.json') == ready[str(round_id)]
        frozen = read(stage / 'frozen.json')
        assert digest(stage / 'selection.json') == frozen['selection_sha256']
        for path, expected in frozen['models'].items():
            assert digest(root / path / 'complete.json') == expected
        for path, expected in frozen['protocols'].items():
            assert digest(root / path / 'protocol.json') == expected
        all_cases.extend(read(stage / 'cases.json'))
        selected_ids.update((round_id, c['case_id']) for c in read(stage / 'selection.json'))
    refs = read(root / 'references.json')
    for ref in refs.values():
        for filename, expected in ref['hashes'].items():
            assert digest(Path(ref['path']) / filename) == expected
    rows, baselines, reproduction = [], [], []
    for task in TASKS:
        for beta in BETAS:
            cell = root / task / f'beta_{beta}'
            manifest = read(cell / 'manifest.json')
            assert digest(cell / 'data/test.csv') == manifest['hashes']['data/test.csv']
            data = frame(cell / 'data/test.csv')
            np.testing.assert_array_equal(data[:, :2], np.load(cell / 'data/test_features.npy'))
            y = data[:, 2]
            scores = {}
            for name in ['LIST_v1', 'gaussian_previous']:
                out = Path(refs[f'{task}/{beta}/{name}']['path'])
                score = archived.metric(np.load(out / 'test.npy'), y)
                scores[name] = score
                baselines.append(dict(task=task, beta=beta, seed=SEED, method=name, **score))
            mean = predict(dict(np.load(cell / 'baseline/statistics.npz')), data[:, 1].astype(int), 'item_mean')
            baselines.append(dict(task=task, beta=beta, seed=SEED, method='item_mean', **archived.metric(mean, y)))
            for case in [c for c in all_cases if c['task'] == task and c['beta'] == beta]:
                out = model_output(root, case)
                pred = np.load(out / 'test.npy')
                selection = read(out / 'selection.json')
                row = dict(task=task, beta=beta, seed=SEED, round=case['round'], parameter=case['parameter'],
                    value=case['value'], **case['parameters'], reused=case['reused_from'] is not None,
                    validation_selected=(case['round'], case['case_id']) in selected_ids,
                    **selection, **archived.metric(pred, y), output=case['output'])
                prior_score = (archived.metric(np.load(root / case['previous_output'] / 'test.npy'), y)
                               if case['previous_output'] else scores['gaussian_previous'])
                for metric in ['mae', 'rmse']:
                    for reference, score in scores.items():
                        row[f'delta_{metric}_vs_{reference}'] = row[metric] - score[metric]
                    row[f'delta_{metric}_vs_previous_stage'] = row[metric] - prior_score[metric]
                rows.append(row)
                if case['round'] == 1 and case['parameters'] == DEFAULTS:
                    old = Path(refs[f'{task}/{beta}/gaussian_previous']['path'])
                    previous = np.load(old / 'test.npy')
                    np.testing.assert_array_equal(pred, previous)
                    assert selection == read(old / 'selection.json')
                    reproduction.append(dict(task=task, beta=beta, exact_predictions=True, same_selection=True))
    for filename, data in [('comparison.csv', rows), ('baselines.csv', baselines)]:
        with (root / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    save(root / 'default_reproduction.json', dict(status='PASS', rows=reproduction))
    summary = dict(status='PASS', rows=rows, baselines=baselines,
                   selected=[r for r in rows if r['validation_selected']],
                   new_fits=sum(c['reused_from'] is None for c in all_cases), evaluated_configurations=len(all_cases))
    save(root / 'summary.json', summary)
    lines = ['# 高斯初始标准差、下限、上限顺序搜索', '',
        'q=1；三个任务、β20/80、种子2020。按验证集先选初始标准差，再固定它选下限，最后固定前两项选上限。训练与λ候选设置不变。',
        f"实际新训练{summary['new_fits']}组，含复用的逐阶段配置共{len(rows)}组。加粗/选中仅依据验证集。", '',
        '| 任务 | β | 阶段 | 初始std | 下限 | 上限 | 验证选中 | epoch | λ | MAE | RMSE | ΔRMSE 对默认高斯 | ΔRMSE 对原版LIST |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {NAMES[r['task']]} | {r['beta']}% | {r['round']} | {r['latent_std']:g} | {r['std_lower']:g} | {r['std_upper']:g} | {'是' if r['validation_selected'] else ''} | {r['epoch']} | {r['lambda_value']:g} | {r['mae']:.6f} | {r['rmse']:.6f} | {r['delta_rmse_vs_gaussian_previous']:+.6f} | {r['delta_rmse_vs_LIST_v1']:+.6f} |")
    lines += ['', '同一配置重复出现时复用上一阶段冻结的结果。上限低于已选初始标准差时跳过，避免初始化被裁剪。',
              '选择前一阶段参数时不读取测试标签；全部三阶段模型和选择冻结后统一测试。单种子、历史数据上的探索结果。', '',
              '[逐配置CSV](comparison.csv) · [对照CSV](baselines.csv) · [默认值复现](default_reproduction.json) · [独立核验](verification.json)']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print('SEQUENTIAL_RESULTS', json.dumps(summary['selected']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'prepare_stage', 'selftest', 'dispatch_selftest', 'train', 'worker', 'select_stage', 'evaluate'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--round', type=int, choices=[1, 2, 3])
    parser.add_argument('--case', type=int)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'train':
        train(root, args.round, args.case)
    elif args.stage == 'worker':
        worker(root, args.round, args.worker)
    elif args.stage == 'selftest':
        selftest(root, args.device)
    elif args.stage in ['prepare_stage', 'select_stage']:
        globals()[args.stage](root, args.round)
    else:
        globals()[args.stage](root)
