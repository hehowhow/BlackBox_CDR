"""One-factor Gaussian rating-noise sweep: three tasks, two betas, seed2020."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from book_movie_protocol import BASE, digest, save, predict
from list_source_backbones import read, check_code, student_guard
from small_task_comparison import frame
from residual_experiment import seed_all
from list_ambiguity_model import ConditionalList
from list_gaussian_noise_model import GaussianNoiseList, joint_nll_twice
import list_ambiguity_story as archived

MODEL_PARENT = BASE / 'runs/list_ambiguity_8913'
TRANSFER_PARENT = BASE / 'runs/list_ambiguity_transfer_8923'
DATA_PARENT = BASE / 'runs/paper_list_dnn_8638'
TASKS = {'music_movie_clean': 17904, 'book_movie': 37388, 'book_music': 16738}
NAMES = {'music_movie_clean': 'Music→Movie', 'book_movie': 'Book→Movie', 'book_music': 'Book→Music'}
BETAS = [20, 80]
SEED = 2020
# Start with q=1; ties in validation retain the original variance.
NOISES = [1., .25, .5, 2., 4.]


def baseline_root(task):
    return BASE / 'runs/small_task_comparison_8650' if task == 'music_movie_clean' else TRANSFER_PARENT / task


def reference_gaussian(task, beta):
    root = MODEL_PARENT if task == 'music_movie_clean' else TRANSFER_PARENT / task
    return root / f'gaussian_learned/beta_{beta}/seed_{SEED}'


def branch(root, task, noise):
    return root / task / ('noise_' + format(float(noise), 'g').replace('.', 'p'))


def output(root, task, noise, beta):
    return branch(root, task, noise) / f'gaussian_learned/beta_{beta}/seed_{SEED}'


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    cfg = read(MODEL_PARENT / 'protocol.json')
    parents = [MODEL_PARENT, TRANSFER_PARENT, DATA_PARENT]
    hashes, parent_hashes = {}, {}
    for parent in parents:
        assert read(parent / 'verification.json')['status'] == 'PASS'
        hashes.update(read(parent / 'protocol.json')['code_hashes'])
        parent_hashes[str(parent)] = digest(parent / 'protocol.json')
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    files = [Path(__file__), BASE / 'scripts/list_gaussian_noise_model.py',
             BASE / 'scripts/verify_list_gaussian_noise.py', BASE / 'scripts/slurm_list_gaussian_noise.sh']
    cfg.update(task=None, tasks=TASKS, seeds=[SEED], seed=SEED, betas=BETAS,
        variants=['gaussian_learned'], observation_variances=NOISES,
        parent_protocol_hashes=parent_hashes,
        changed_parameter='only fixed observation/rating-noise variance q; q is variance, not std',
        unchanged='10 epochs; Adam .001; wd1e-5; 64 complete users/batch; dim32; hidden64; dropout.1; latent std init.3/bounds.05..2; DNN Top100; ordinary means; lambda grid 0/.25/.5/.75/1; selection and fresh refit',
        objective='exact joint Gaussian marginal: q I + V diag(var) V^T; 2*NLL excluding n*log(2*pi), including n*log(q); sum user losses / batch rating count; no extra loss rescaling',
        evaluation='30 fits/predictions and validation-only noise choices frozen before test evaluation; compare same seed2020 archived LIST and Gaussian; historical split, exploratory',
        code_hashes={**hashes, **{str(p.resolve()): digest(p) for p in files}})
    save(root / 'protocol.json', cfg)
    refs, source_records = {}, {}
    for task, overlap in TASKS.items():
        hub = root / task
        hub.mkdir()
        query = DATA_PARENT / task / f'seed_{SEED}/queries/DNN'
        record = read(query / 'complete.json')
        assert record['status'] == 'PASS' and record['k'] == 100 and record['users'] == overlap
        assert digest(query / 'topk_ids.npz') == record['item_ids_sha256']
        ids = np.load(query / 'topk_ids.npz')
        np.testing.assert_array_equal(ids['user_ids'], np.arange(overlap))
        assert ids['item_ids'].shape == (overlap, 100)
        source_records[task] = dict(path=str(query / 'topk_ids.npz'), sha256=record['item_ids_sha256'])
        for beta in BETAS:
            old = baseline_root(task) / f'beta_{beta}'
            cell = hub / f'beta_{beta}'
            cell.mkdir()
            manifest = read(old / 'manifest.json')
            assert manifest['overlap_users'] == overlap and manifest['cross_domain_raw_item_overlap'] == 0
            for path, expected in manifest['hashes'].items():
                if path != 'data/test.csv':
                    assert digest(old / path) == expected, path
            for name in ['data', 'baseline', 'inner_baseline']:
                (cell / name).symlink_to((old / name).resolve(), target_is_directory=True)
            save(cell / 'manifest.json', manifest)
            for variant, ref in [('LIST_v1', old / f'seed_{SEED}/residual_real'),
                                 ('gaussian_previous', reference_gaussian(task, beta))]:
                done = read(ref / 'complete.json')
                assert done['status'] == 'PASS'
                refs[f'{task}/{beta}/{variant}'] = dict(path=str(ref),
                    hashes={n: digest(ref / n) for n in ['complete.json', 'test.npy', 'selection.json']})
        for noise in NOISES:
            sub = branch(root, task, noise)
            sub.mkdir()
            save(sub / 'protocol.json', dict(cfg, task=task, observation_variance=noise))
            for beta in BETAS:
                cell = sub / f'beta_{beta}'
                cell.mkdir()
                for name in ['data', 'baseline', 'inner_baseline', 'manifest.json']:
                    (cell / name).symlink_to((hub / f'beta_{beta}' / name).resolve(),
                                           target_is_directory=name != 'manifest.json')
                output(root, task, noise, beta).mkdir(parents=True)
    tasks = [dict(task=task, noise=noise, beta=beta) for noise in NOISES for task in TASKS for beta in BETAS]
    save(root / 'tasks.json', tasks)
    save(root / 'references.json', refs)
    save(root / 'source_records.json', source_records)
    save(root / 'data_verification.json', dict(status='PASS', models=len(tasks),
         only_observation_variance_changes=True, no_test_labels_opened=True))
    print('GAUSSIAN_NOISE_PREPARED', root, len(tasks), flush=True)


def selftest(root, device):
    check_code(root)
    student_guard(root)
    cfg = read(root / 'protocol.json')
    # Independent full rating-covariance calculation, including unequal-length
    # histories; compare values AND gradients with the efficient latent formula.
    seed_all(731)
    v = torch.randn(2, 5, 3, device=device, dtype=torch.float64, requires_grad=True)
    mean = torch.randn(2, 3, device=device, dtype=torch.float64, requires_grad=True)
    logvar = torch.randn(2, 3, device=device, dtype=torch.float64, requires_grad=True)
    r = torch.randn(2, 5, device=device, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], device=device, dtype=torch.float64)
    for q in NOISES:
        vm, rm = v * mask[..., None], r * mask
        a, b, c = vm.transpose(1, 2) @ vm, (vm * rm[..., None]).sum(1), rm.square().sum(1)
        actual = joint_nll_twice(mean, logvar.exp(), a, b, c, mask.sum(-1), q)
        references = []
        for u, n in enumerate([5, 3]):
            vu = v[u, :n]
            covariance = q * torch.eye(n, device=device, dtype=v.dtype) + (vu * logvar[u].exp()) @ vu.T
            distribution = torch.distributions.MultivariateNormal(vu @ mean[u], covariance_matrix=covariance)
            references.append(-2 * distribution.log_prob(r[u, :n]) - n * math.log(2 * math.pi))
        expected = torch.stack(references)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        grad_actual = torch.autograd.grad(actual.sum(), (mean, logvar, v), retain_graph=True)
        grad_expected = torch.autograd.grad(expected.sum(), (mean, logvar, v), retain_graph=True)
        for ga, ge in zip(grad_actual, grad_expected):
            torch.testing.assert_close(ga, ge, atol=1e-9, rtol=1e-9)
    # q=1 must have identical model initialization, loss, gradients and inference.
    lists = torch.tensor([[1, 2, 3], [3, 2, 1], [0, 0, 0]], device=device)
    base = torch.tensor([3., 4., 2.], device=device)
    seed_all(123)
    original = ConditionalList(lists, 3, 3, base, cfg, 'gaussian_learned').to(device)
    seed_all(123)
    revised = GaussianNoiseList(lists, 3, 3, base, cfg, 'gaussian_learned').to(device)
    assert archived.parameter_hash(original) == archived.parameter_hash(revised)
    with torch.no_grad():
        original.encoder[-1].weight.normal_(std=.1)
        revised.load_state_dict(original.state_dict())
    users = torch.tensor([0, 1], device=device)
    items = torch.tensor([[0, 1, 2], [1, 2, 0]], device=device)
    residual = torch.tensor([[1., -.3, .5], [-.2, 1., 0.]], device=device)
    mask32 = torch.tensor([[1., 1., 1.], [1., 1., 0.]], device=device)
    losses = []
    for model in [original, revised]:
        seed_all(911)
        loss, _ = model.objective(users, items, residual, mask32, 42)
        loss.backward()
        losses.append(loss)
        model.eval()
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    for (na, pa), (nb, pb) in zip(original.named_parameters(), revised.named_parameters()):
        assert na == nb
        if pa.grad is None:
            assert pb.grad is None
        else:
            torch.testing.assert_close(pa.grad, pb.grad, atol=0, rtol=0)
    torch.testing.assert_close(original.user_mean(torch.arange(3, device=device), 42),
                               revised.user_mean(torch.arange(3, device=device), 42), atol=0, rtol=0)
    save(root / f'selftest_{device}.json', dict(status='PASS', noises=NOISES,
        full_covariance_values_and_gradients=True, unequal_history_lengths=True,
        exact_q1_loss_gradients_initialization_inference=True))
    print('GAUSSIAN_NOISE_SELFTEST_PASS', device, flush=True)


def train(root, task, noise, beta):
    check_code(root)
    sub = branch(root, task, noise)
    assert read(sub / 'protocol.json')['observation_variance'] == noise
    archived.ConditionalList = GaussianNoiseList
    archived.PRIOR = DATA_PARENT
    archived.TASK = task
    archived.PARENT = baseline_root(task)
    archived.train(sub, 'gaussian_learned', beta, SEED)


def worker(root, index):
    for task in read(root / 'tasks.json'):
        out = output(root, **task)
        try:
            with (out / 'claim.json').open('x') as handle:
                json.dump(dict(worker=index, gpu=os.environ['CUDA_VISIBLE_DEVICES']), handle)
        except FileExistsError:
            continue
        print('GAUSSIAN_NOISE_TASK', index, task, flush=True)
        command = [sys.executable, '-u', str(Path(__file__)), 'train', '--root', str(root)]
        for key, value in task.items():
            command += ['--' + key, str(value)]
        with (out / 'train.log').open('w') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def freeze(root):
    check_code(root)
    assert read(root / 'selftest_cuda.json')['status'] == 'PASS'
    # This stage has a test-label access guard: noise is selected on validation.
    student_guard(root)
    frozen = {}
    for task in read(root / 'tasks.json'):
        out = output(root, **task)
        done = read(out / 'complete.json')
        assert done['status'] == 'PASS' and done['test_labels_opened'] is False
        for name, key in [('final.pt', 'final_sha256'), ('test.npy', 'test_sha256'), ('selection.json', 'selection_sha256')]:
            assert digest(out / name) == done[key]
        frozen[str(out.relative_to(root))] = digest(out / 'complete.json')
    assert len(frozen) == 30
    choices = []
    for task in TASKS:
        for beta in BETAS:
            selections = [(q, read(output(root, task, q, beta) / 'selection.json')) for q in NOISES]
            q, selected = min(selections, key=lambda pair: pair[1]['validation_rmse'])
            choices.append(dict(task=task, beta=beta, noise=q, **selected))
    with (root / 'noise_selection.json').open('x') as handle:
        json.dump(choices, handle, indent=2)
    with (root / 'evaluation_ready.json').open('x') as handle:
        json.dump(dict(models=frozen, selection_sha256=digest(root / 'noise_selection.json')), handle, indent=2)
    print('GAUSSIAN_NOISE_FROZEN', choices, flush=True)


def evaluate(root):
    check_code(root)
    ready = read(root / 'evaluation_ready.json')
    assert digest(root / 'noise_selection.json') == ready['selection_sha256']
    for rel, expected in ready['models'].items():
        assert digest(root / rel / 'complete.json') == expected
    refs = read(root / 'references.json')
    for ref in refs.values():
        for name, expected in ref['hashes'].items():
            assert digest(Path(ref['path']) / name) == expected
    choices = {(r['task'], r['beta']): r['noise'] for r in read(root / 'noise_selection.json')}
    rows, replay = [], []
    for task in TASKS:
        for beta in BETAS:
            cell = root / task / f'beta_{beta}'
            manifest = read(cell / 'manifest.json')
            assert digest(cell / 'data/test.csv') == manifest['hashes']['data/test.csv']
            data = frame(cell / 'data/test.csv')
            np.testing.assert_array_equal(data[:, :2], np.load(cell / 'data/test_features.npy'))
            y = data[:, 2]
            reference_predictions = {name: np.load(Path(refs[f'{task}/{beta}/{name}']['path']) / 'test.npy')
                                     for name in ['LIST_v1', 'gaussian_previous']}
            scores = {name: archived.metric(pred, y) for name, pred in reference_predictions.items()}
            current = np.load(output(root, task, 1., beta) / 'test.npy')
            old = reference_predictions['gaussian_previous']
            np.testing.assert_allclose(current, old, atol=1e-7, rtol=0)
            old_selection = read(Path(refs[f'{task}/{beta}/gaussian_previous']['path']) / 'selection.json')
            new_selection = read(output(root, task, 1., beta) / 'selection.json')
            assert new_selection == old_selection, (task, beta, new_selection, old_selection)
            replay.append(dict(task=task, beta=beta, exact_predictions=bool(np.array_equal(current, old)),
                               max_prediction_difference=float(np.max(np.abs(current - old))), same_selection=True))
            candidates = [('LIST_v1', None), ('gaussian_previous', 1.),
                          ('item_mean', None), *[('gaussian_noise', q) for q in sorted(NOISES)]]
            for name, q in candidates:
                if name in reference_predictions:
                    pred = reference_predictions[name]
                    selected = read(Path(refs[f'{task}/{beta}/{name}']['path']) / 'selection.json')
                elif name == 'item_mean':
                    pred = predict(dict(np.load(cell / 'baseline/statistics.npz')), data[:, 1].astype(int), 'item_mean')
                    selected = dict(epoch=0, lambda_value=0., validation_rmse=None)
                else:
                    out = output(root, task, q, beta)
                    pred = np.load(out / 'test.npy')
                    selected = read(out / 'selection.json')
                row = dict(task=task, beta=beta, seed=SEED, method=name, observation_variance=q,
                           validation_selected=name == 'gaussian_noise' and q == choices[(task, beta)],
                           **selected, **archived.metric(pred, y))
                for metric in ['mae', 'rmse']:
                    for ref in ['LIST_v1', 'gaussian_previous']:
                        row[f'delta_{metric}_vs_{ref}'] = row[metric] - scores[ref][metric]
                rows.append(row)
    with (root / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(root / 'q1_reproduction.json', dict(status='PASS', rows=replay))
    save(root / 'summary.json', dict(status='PASS', rows=rows, noise_selection=read(root / 'noise_selection.json')))
    lines = ['# 高斯评分噪声方差：单参数搜索', '',
             '三个任务、β20/80、仅种子2020。q是评分噪声方差，搜索0.25/0.5/1/2/4。其他训练、模型及λ候选参数全部不变。',
             '原版LIST及旧高斯均使用同一seed2020历史结果；新跑q=1复核旧结果。按原规则验证选epoch/λ、从头外层重训；再根据验证RMSE选择q。所有30个模型和q选择在测试评估前冻结。', '',
             '| 任务 | β | 方法 | q | 验证选中q | epoch | λ | MAE | RMSE | ΔRMSE 对旧高斯 | ΔRMSE 对原版LIST |',
             '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    labels = {'LIST_v1': '原版LIST', 'gaussian_previous': '旧高斯', 'item_mean': '普通均值', 'gaussian_noise': '高斯搜索'}
    for r in rows:
        lines.append(f"| {NAMES[r['task']]} | {r['beta']}% | {labels[r['method']]} | {r['observation_variance'] if r['observation_variance'] is not None else '—'} | {'是' if r['validation_selected'] else ''} | {r['epoch']} | {r['lambda_value']} | {r['mae']:.6f} | {r['rmse']:.6f} | {r['delta_rmse_vs_gaussian_previous']:+.6f} | {r['delta_rmse_vs_LIST_v1']:+.6f} |")
    lines += ['', '差值为当前方法减对照，负值更好。只有一个种子，不报告标准差或跨种子稳定性结论。已复用历史测试集，属于探索性实验。',
              'λ候选集合及训练设置没有调大；不同q可按原验证规则选中不同epoch/λ。每个用户全部训练评分进入联合似然；q同时改变该似然的拟合项和协方差项，不额外乘损失缩放系数。', '',
              '[完整CSV](comparison.csv) · [验证集选择](noise_selection.json) · [默认值复现](q1_reproduction.json) · [独立复核](verification.json) · [协议](protocol.json)']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print('GAUSSIAN_NOISE_RESULTS', json.dumps(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'selftest', 'train', 'worker', 'freeze', 'evaluate'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--noise', type=float, choices=NOISES)
    parser.add_argument('--beta', type=int, choices=BETAS)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'train':
        train(root, args.task, args.noise, args.beta)
    elif args.stage == 'worker':
        worker(root, args.worker)
    elif args.stage == 'selftest':
        selftest(root, args.device)
    else:
        globals()[args.stage](root)
