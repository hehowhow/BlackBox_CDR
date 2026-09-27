"""Six independent sensitivity groups; train/eval step sizes are separate axes."""
import argparse
import csv
import json
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
import list_ambiguity_model as original_model
import list_ambiguity_story as archived
from list_langevin_sensitivity_model import SplitSamplerList
from list_gaussian_noise import (TASKS, NAMES, BETAS, SEED, DATA_PARENT,
                                 MODEL_PARENT, TRANSFER_PARENT, baseline_root)

PARENT = BASE / 'runs/list_gaussian_noise_8952'
DEFAULTS = dict(train_mala_step=.25, eval_mala_step=.25, train_steps=20,
                eval_steps=20, train_chains=32, eval_chains=128, energy_bound=5.)
# Six groups as in the earlier parameter table. Splitting its shared step size
# produces two independent one-factor axes inside group1, not a Cartesian grid.
AXES = {
    'train_mala_step': dict(group=1, values=[.25, .1, .5], label='训练步长'),
    'eval_mala_step': dict(group=1, values=[.25, .1, .5], label='预测步长'),
    'train_steps': dict(group=2, values=[20, 10, 60], label='训练采样步数'),
    'eval_steps': dict(group=3, values=[20, 10, 60], label='预测采样步数'),
    'train_chains': dict(group=4, values=[32, 16, 64], label='训练链数'),
    'eval_chains': dict(group=5, values=[128, 64, 256], label='预测链数'),
    'energy_bound': dict(group=6, values=[5., 1., 10.], label='能量修正上界'),
}


def output(root, case):
    return root / case['output']


def reference_energy(task, beta):
    parent = MODEL_PARENT if task == 'music_movie_clean' else TRANSFER_PARENT / task
    return parent / f'energy_20/beta_{beta}/seed_{SEED}'


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    assert read(PARENT / 'verification.json')['status'] == 'PASS'
    cfg = read(PARENT / 'protocol.json')
    cfg.pop('observation_variances')
    cfg.update(**DEFAULTS, variants=['energy_20'], defaults=DEFAULTS, axes=AXES,
        sensitivity_parent=str(PARENT), sensitivity_parent_sha256=digest(PARENT / 'protocol.json'),
        changed_parameter='one scalar only per case; group1 has separate train/eval step-size subscans; six groups, seven axes',
        unchanged='archived energy_20; latent init.3/std bounds.05..2; noise variance1; dim32/hidden64/dropout.1; 10 epochs; Adam.001/wd1e-5; 64 users/batch; DNN Top100; ordinary means; original lambda grid and fresh refit',
        gaussian_policy='Gaussian future bounds .05..1 do NOT change this energy-model control; all energy defaults reproduce the archived energy_20',
        objective='byte-identical archived conditional energy objective and MALA sampler; only phase-specific controls supplied to archived calls',
        selection='original validation-only epoch/lambda selection; all candidate values fixed upfront; each axis resets other coordinates to DEFAULTS; no sequential optimization',
        evaluation='freeze all90 models and predictions before test labels; report every candidate and paired same-seed deltas; default model shared across all axes; one seed2020, exploratory',
        sampling='MALA with Metropolis correction; train prior/posterior share train step size and chain length; inference has independent step size and chain length',
        hidden_width_fixed=64)
    for path, expected in cfg['code_hashes'].items():
        assert digest(Path(path)) == expected, path
    files = [Path(__file__), BASE / 'scripts/list_langevin_sensitivity_model.py',
             BASE / 'scripts/verify_list_langevin_sensitivity.py',
             BASE / 'scripts/slurm_list_langevin_sensitivity.sh']
    cfg['code_hashes'].update({str(p.resolve()): digest(p) for p in files})
    save(root / 'protocol.json', cfg)
    source_records = read(PARENT / 'source_records.json')
    for record in source_records.values():
        assert digest(Path(record['path'])) == record['sha256']
    save(root / 'source_records.json', source_records)
    refs = {}
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
            for name, ref in [('LIST_v1', baseline_root(task) / f'beta_{beta}/seed_{SEED}/residual_real'),
                              ('energy_previous', reference_energy(task, beta))]:
                assert read(ref / 'complete.json')['status'] == 'PASS'
                refs[f'{task}/{beta}/{name}'] = dict(path=str(ref),
                    hashes={n: digest(ref / n) for n in ['complete.json', 'test.npy', 'selection.json']})
    save(root / 'references.json', refs)
    settings = [('default', None, 0, DEFAULTS)]
    for key, axis in AXES.items():
        for value in axis['values'][1:]:
            settings.append((key, value, axis['group'], dict(DEFAULTS, **{key: value})))
    cases = []
    for key, value, group, controls in settings:
        tag = 'default' if key == 'default' else key + '_' + format(value, 'g').replace('.', 'p')
        for task in TASKS:
            branch = root / 'models' / tag / task
            branch.mkdir(parents=True)
            save(branch / 'protocol.json', dict(cfg, task=task, **controls))
            for beta in BETAS:
                cell = branch / f'beta_{beta}'
                cell.mkdir()
                for name in ['data', 'baseline', 'inner_baseline', 'manifest.json']:
                    (cell / name).symlink_to((root / task / f'beta_{beta}' / name).resolve(),
                                           target_is_directory=name != 'manifest.json')
                out = branch / f'energy_20/beta_{beta}/seed_{SEED}'
                out.mkdir(parents=True)
                cases.append(dict(case_id=len(cases), task=task, beta=beta, group=group,
                    axis=key, value=value, controls=controls,
                    branch=str(branch.relative_to(root)), output=str(out.relative_to(root))))
    assert len(cases) == 90
    save(root / 'cases.json', cases)
    save(root / 'data_verification.json', dict(status='PASS', cases_sha256=digest(root / 'cases.json'),
         models=len(cases), no_test_labels_opened=True, all_candidates_fixed_upfront=True))
    print('LANGEVIN_PREPARED', root, len(cases), flush=True)


def selftest(root, device):
    check_code(root)
    student_guard(root)
    cfg = read(root / 'protocol.json')
    lists = torch.tensor([[1, 2, 3], [3, 2, 1], [0, 0, 0]], device=device)
    base = torch.tensor([3., 4., 2.], device=device)
    users = torch.arange(3, device=device)
    items = torch.tensor([[0, 1, 2], [1, 2, 0]], device=device)
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]], device=device)
    residual = torch.tensor([[1., -.3, .5], [-.2, 1., 0.]], device=device)
    seed_all(SEED)
    old = original_model.ConditionalList(lists, 3, 3, base, cfg, 'energy_20').to(device)
    seed_all(SEED)
    new = SplitSamplerList(lists, 3, 3, base, cfg, 'energy_20').to(device)
    assert archived.parameter_hash(old) == archived.parameter_hash(new)
    # Nonzero tilt and mean ensure parity covers nontrivial accepted/rejected
    # proposals, not only the zero-output initialization.
    with torch.no_grad():
        old.encoder[-1].weight.normal_(std=.1)
        old.energy_net[-1].weight.normal_(std=.15)
        old.energy_net[-1].bias.fill_(.1)
    new.load_state_dict(old.state_dict())

    def loss_grads(model):
        model.train()
        model.zero_grad(set_to_none=True)
        seed_all(147)
        loss, extra = model.objective(users[:2], items, residual, mask, 789)
        loss.backward()
        gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        return loss.detach(), gradients, extra

    a, ga, ea = loss_grads(old)
    b, gb, eb = loss_grads(new)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert ga.keys() == gb.keys()
    for key in ga:
        torch.testing.assert_close(ga[key], gb[key], atol=0, rtol=0)
    for key in ea:
        torch.testing.assert_close(ea[key], eb[key], atol=0, rtol=0)
    old.eval()
    new.eval()
    inference = new.user_mean(users, 123)
    torch.testing.assert_close(old.user_mean(users, 123), inference, atol=0, rtol=0)
    assert inference[-1].abs().max() == 0

    # Changing inference controls must leave training loss AND every gradient
    # unchanged; changing train controls must leave fixed-model inference equal.
    new.cfg = dict(cfg, eval_steps=3, eval_mala_step=.7, eval_chains=8)
    c, gc, _ = loss_grads(new)
    torch.testing.assert_close(c, b, atol=0, rtol=0)
    for key in gb:
        torch.testing.assert_close(gb[key], gc[key], atol=0, rtol=0)
    new.cfg = dict(cfg, train_steps=3, train_mala_step=.7, train_chains=8)
    new.eval()
    torch.testing.assert_close(new.user_mean(users, 123), inference, atol=0, rtol=0)

    # Spy on the actual sampler, exercising both prior and posterior dispatch.
    original_sampler = original_model.langevin
    calls = []
    def spy(*args):
        calls.append((args[4], args[5], args[6]))
        return original_sampler(*args)
    original_model.langevin = spy
    new.cfg = dict(cfg, train_steps=3, eval_steps=5, train_mala_step=.1,
                   eval_mala_step=.5, train_chains=4, eval_chains=6)
    snapshot = dict(new.cfg)
    try:
        new.objective(users[:2], items, residual, mask, 123)
        new.user_mean(users, 123)
    finally:
        original_model.langevin = original_sampler
    assert calls == [(3, 4, .1), (3, 4, .1), (5, 6, .5)], calls
    assert new.cfg == snapshot and new.steps == 20
    assert cfg == read(root / 'protocol.json'), 'shared configuration was mutated'
    save(root / f'selftest_{device}.json', dict(status='PASS', exact_default_initialization=True,
        exact_default_nonzero_energy_loss_gradients_and_inference=True,
        inference_controls_do_not_change_training=True, training_controls_do_not_change_inference=True,
        actual_sampler_dispatch=calls, no_added_parameters=True, hidden_width=64))
    print('LANGEVIN_SELFTEST_PASS', device, flush=True)


def gpu_available(device):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu', '--format=csv'],
                            check=True, capture_output=True, text=True)
    for row in csv.DictReader(result.stdout.splitlines()):
        row = {key.strip(): value.strip() for key, value in row.items()}
        if row['index'] == device:
            used = float(row['memory.used [MiB]'].split()[0])
            free = float(row['memory.free [MiB]'].split()[0])
            utilization = float(row['utilization.gpu [%]'].split()[0])
            return used <= 512 or (free >= 8192 and utilization <= 5)
    raise RuntimeError(f'Allocated GPU {device} was not found')


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


def train(root, case_id):
    check_code(root)
    case = read(root / 'cases.json')[case_id]
    assert case['case_id'] == case_id
    archived.ConditionalList = SplitSamplerList
    archived.PRIOR = DATA_PARENT
    archived.TASK = case['task']
    archived.PARENT = baseline_root(case['task'])
    archived.train(root / case['branch'], 'energy_20', case['beta'], SEED)


def worker(root, group, index):
    cases = [c for c in read(root / 'cases.json') if c['group'] == group]
    available = False
    for case in cases:
        out = output(root, case)
        if (out / 'claim.json').exists():
            continue
        while not available:
            if all((output(root, c) / 'claim.json').exists() for c in cases):
                return
            if gpu_available(os.environ['CUDA_VISIBLE_DEVICES']):
                available = True
            else:
                print('WAIT_ALLOCATED_GPU_IDLE', os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
                time.sleep(10)
        try:
            with (out / 'claim.json').open('x') as handle:
                json.dump(dict(worker=index, gpu=os.environ['CUDA_VISIBLE_DEVICES'], time=time.time()), handle)
        except FileExistsError:
            continue
        command = [sys.executable, '-u', str(Path(__file__)), 'train', '--root', str(root), '--case', str(case['case_id'])]
        print('LANGEVIN_CASE', index, case, flush=True)
        with (out / 'train.log').open('w') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def freeze_group(root, group):
    check_code(root)
    student_guard(root)
    cases = [c for c in read(root / 'cases.json') if c['group'] == group]
    frozen, replays = {}, []
    for case in cases:
        out = output(root, case)
        done = read(out / 'complete.json')
        assert done['status'] == 'PASS' and done['test_labels_opened'] is False
        for name, key in [('final.pt', 'final_sha256'), ('test.npy', 'test_sha256'), ('selection.json', 'selection_sha256')]:
            assert digest(out / name) == done[key]
        frozen[case['output']] = digest(out / 'complete.json')
        if group == 0:
            ref = reference_energy(case['task'], case['beta'])
            current, previous = np.load(out / 'test.npy'), np.load(ref / 'test.npy')
            np.testing.assert_allclose(current, previous, atol=1e-7, rtol=0)
            assert read(out / 'selection.json') == read(ref / 'selection.json')
            replays.append(dict(task=case['task'], beta=case['beta'], exact_predictions=bool(np.array_equal(current, previous)),
                                max_abs_difference=float(np.max(np.abs(current - previous)))))
    save(root / f'group_{group}_frozen.json', dict(status='PASS', models=frozen, time=time.time()))
    if group == 0:
        save(root / 'default_reproduction.json', dict(status='PASS', rows=replays))
    print('LANGEVIN_GROUP_FROZEN', group, len(cases), flush=True)


def freeze(root):
    check_code(root)
    student_guard(root)
    assert read(root / 'selftest_cuda.json')['status'] == 'PASS'
    assert read(root / 'default_reproduction.json')['status'] == 'PASS'
    models = {}
    for group in range(7):
        frozen = read(root / f'group_{group}_frozen.json')
        for rel, expected in frozen['models'].items():
            assert digest(root / rel / 'complete.json') == expected
            models[rel] = expected
    assert len(models) == 90
    cases = read(root / 'cases.json')
    choices = []
    for task in TASKS:
        for beta in BETAS:
            baseline = next(c for c in cases if c['task'] == task and c['beta'] == beta and c['axis'] == 'default')
            for key, axis in AXES.items():
                candidates = [baseline] + [c for c in cases if c['task'] == task and c['beta'] == beta and c['axis'] == key]
                best = min(candidates, key=lambda c: read(output(root, c) / 'selection.json')['validation_rmse'])
                choices.append(dict(task=task, beta=beta, axis=key, value=best['controls'][key], case_id=best['case_id']))
    save(root / 'validation_choices.json', choices)
    with (root / 'evaluation_ready.json').open('x') as handle:
        json.dump(dict(models=models, choices_sha256=digest(root / 'validation_choices.json'), time=time.time()), handle, indent=2)
    print('LANGEVIN_ALL_FROZEN', len(models), flush=True)


def evaluate(root):
    check_code(root)
    ready = read(root / 'evaluation_ready.json')
    assert digest(root / 'validation_choices.json') == ready['choices_sha256']
    for rel, expected in ready['models'].items():
        assert digest(root / rel / 'complete.json') == expected
    references = read(root / 'references.json')
    for ref in references.values():
        for name, expected in ref['hashes'].items():
            assert digest(Path(ref['path']) / name) == expected
    choices = {(r['task'], r['beta'], r['axis']): r for r in read(root / 'validation_choices.json')}
    cases = read(root / 'cases.json')
    rows, baseline_rows, sensitivity = [], [], []
    for task in TASKS:
        for beta in BETAS:
            cell = root / task / f'beta_{beta}'
            manifest = read(cell / 'manifest.json')
            assert digest(cell / 'data/test.csv') == manifest['hashes']['data/test.csv']
            data = frame(cell / 'data/test.csv')
            np.testing.assert_array_equal(data[:, :2], np.load(cell / 'data/test_features.npy'))
            y = data[:, 2]
            baseline = next(c for c in cases if c['task'] == task and c['beta'] == beta and c['axis'] == 'default')
            default_score = archived.metric(np.load(output(root, baseline) / 'test.npy'), y)
            list_score = archived.metric(np.load(Path(references[f'{task}/{beta}/LIST_v1']['path']) / 'test.npy'), y)
            mean_score = archived.metric(predict(dict(np.load(cell / 'baseline/statistics.npz')), data[:, 1].astype(int), 'item_mean'), y)
            baseline_rows.append(dict(task=task, beta=beta, seed=SEED,
                list_mae=list_score['mae'], list_rmse=list_score['rmse'],
                energy_mae=default_score['mae'], energy_rmse=default_score['rmse'],
                mean_mae=mean_score['mae'], mean_rmse=mean_score['rmse']))
            for key, axis in AXES.items():
                axis_rows = []
                for value in axis['values']:
                    case = baseline if value == DEFAULTS[key] else next(c for c in cases if c['task'] == task and c['beta'] == beta and c['axis'] == key and c['value'] == value)
                    out = output(root, case)
                    score = archived.metric(np.load(out / 'test.npy'), y)
                    selected = read(out / 'selection.json')
                    row = dict(task=task, beta=beta, seed=SEED, group=axis['group'], axis=key, value=value,
                        is_default=value == DEFAULTS[key], case_id=case['case_id'],
                        validation_selected=choices[(task, beta, key)]['case_id'] == case['case_id'],
                        **score, **selected)
                    for metric in ['mae', 'rmse']:
                        row[f'delta_{metric}_vs_default'] = score[metric] - default_score[metric]
                        row[f'delta_{metric}_vs_LIST_v1'] = score[metric] - list_score[metric]
                    axis_rows.append(row)
                    rows.append(row)
                sensitivity.append(dict(task=task, beta=beta, axis=key, group=axis['group'],
                    mae_range=max(r['mae'] for r in axis_rows) - min(r['mae'] for r in axis_rows),
                    rmse_range=max(r['rmse'] for r in axis_rows) - min(r['rmse'] for r in axis_rows),
                    max_abs_delta_mae=max(abs(r['delta_mae_vs_default']) for r in axis_rows),
                    max_abs_delta_rmse=max(abs(r['delta_rmse_vs_default']) for r in axis_rows),
                    validation_selected_value=choices[(task, beta, key)]['value']))
    for name, data in [('results', rows), ('baselines', baseline_rows), ('sensitivity', sensitivity)]:
        with (root / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    save(root / 'summary.json', dict(status='PASS', rows=rows, baselines=baseline_rows,
         sensitivity=sensitivity, choices=read(root / 'validation_choices.json')))
    lines = ['# 朗之万六组参数：独立单因素扫描', '',
        '三个任务、β20/80、seed2020。每个候选仅改变一个参数，其他恢复原能量模型默认值；没有串联最优参数。步长组分训练和预测两条轴，合计七条轴、90个独立训练配置。全部模型和验证选择冻结后再统一读取测试标签。', '',
        '训练上限10轮、Adam .001、weight decay1e-5、64用户/批、hidden64、dim32、dropout0.1；λ候选0/.25/.5/.75/1及选模/重训规则不变。能量模型的高斯参考保持初始std0.3、std范围0.05～2、评分噪声方差1。后续高斯0.05～1政策不混入本轮能量扫描。', '',
        '默认：训练/预测步长各0.25，步数各20，训练链32，预测链128，能量上界5。训练步长同时用于训练先验和后验。仅增加独立控制接口，原目标函数和MALA实现直接复用，默认输出精确复核。', '',
        '## 同种子对照', '', '| 任务 | β | 普通均值 MAE / RMSE | 原版LIST MAE / RMSE | 默认能量 MAE / RMSE |', '|---|---|---|---|---|']
    for r in baseline_rows:
        lines.append(f"| {NAMES[r['task']]} | {r['beta']}% | {r['mean_mae']:.6f} / {r['mean_rmse']:.6f} | {r['list_mae']:.6f} / {r['list_rmse']:.6f} | {r['energy_mae']:.6f} / {r['energy_rmse']:.6f} |")
    lines += ['', '## 各轴测试RMSE的最大变化幅度', '',
        '数值为该轴三个候选中，相对默认值的最大绝对变化；不是改进量，变差也计入。不把六档平均成一个改进指标。', '',
        '| 参数 | Music→Movie β20 | Music→Movie β80 | Book→Movie β20 | Book→Movie β80 | Book→Music β20 | Book→Music β80 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key, axis in AXES.items():
        values = [next(s['max_abs_delta_rmse'] for s in sensitivity if s['task'] == task and s['beta'] == beta and s['axis'] == key) for task in TASKS for beta in BETAS]
        lines.append('| ' + axis['label'] + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    for key, axis in AXES.items():
        lines += ['', f"## 第{axis['group']}组：{axis['label']}", '',
            '| 任务 | β | 参数值 | MAE | RMSE | ΔMAE 对默认 | ΔRMSE 对默认 | ΔRMSE 对LIST | epoch | λ | 验证选中 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in rows:
            if r['axis'] != key:
                continue
            value = format(r['value'], 'g') + ('（默认）' if r['is_default'] else '')
            chosen = '是' if r['validation_selected'] else ''
            lines.append(f"| {NAMES[r['task']]} | {r['beta']}% | {value} | {r['mae']:.6f} | {r['rmse']:.6f} | {r['delta_mae_vs_default']:+.6f} | {r['delta_rmse_vs_default']:+.6f} | {r['delta_rmse_vs_LIST_v1']:+.6f} | {r['epoch']} | {r['lambda_value']:g} | {chosen} |")
    lines += ['', '差值为候选减对照，负值更好。每个候选沿用同一验证选模程序，因此选中的epoch、λ可以不同；没有增加训练预算或修正强度搜索范围。若λ=0或epoch=0，最终退回普通均值，输出相同不能解释为采样轨迹完全相同。',
        '敏感性只针对已测的三档范围、单随机种子和历史划分，不代表理论敏感度或统计显著性。有限MALA步数不保证充分混合。本轮不改变模型结构，不组合多个最优值。', '',
        '[完整CSV](results.csv) · [敏感性CSV](sensitivity.csv) · [固定对照](baselines.csv) · [协议](protocol.json) · [默认复现](default_reproduction.json) · [独立复核](verification.json)']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print('LANGEVIN_EVALUATED', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'selftest', 'dispatch_selftest', 'train', 'worker', 'freeze_group', 'freeze', 'evaluate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--group', type=int, choices=range(7))
    parser.add_argument('--worker', type=int)
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'selftest':
        selftest(root, args.device)
    elif args.stage == 'train':
        train(root, args.case)
    elif args.stage == 'worker':
        worker(root, args.group, args.worker)
    elif args.stage == 'freeze_group':
        freeze_group(root, args.group)
    else:
        globals()[args.stage](root)
