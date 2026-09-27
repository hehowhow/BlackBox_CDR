"""Transfer the unchanged 8913 conditional models to Book->Movie/Book->Music.

This driver changes dataset dispatch only. It imports the frozen 8650 LIST
training function and the frozen 8913 conditional-model training function.
Each training task runs in a fresh process; task-specific module data paths
are set explicitly before calling the original, byte-identical functions.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from book_movie_protocol import BASE, digest, save, predict
from list_source_backbones import read, check_code
from small_task_comparison import frame
import small_task_comparison as archived
import list_ambiguity_story as conditional

DATA_PARENT = BASE / 'runs/paper_list_dnn_8638'
MODEL_PARENT = BASE / 'runs/list_ambiguity_8913'
TASKS = {'book_movie': 37388, 'book_music': 16738}
SEEDS = [2020, 2021, 2022]
BETAS = [20, 80]
VARIANTS = ['LIST_v1', 'point_matched', 'gaussian_learned', 'energy_20']


def output(root, task, variant, beta, seed):
    if variant == 'LIST_v1':
        return root / task / f'beta_{beta}/seed_{seed}/residual_real'
    return root / task / variant / f'beta_{beta}/seed_{seed}'


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    model_cfg = read(MODEL_PARENT / 'protocol.json')
    data_cfg = read(DATA_PARENT / 'protocol.json')
    assert read(MODEL_PARENT / 'verification.json')['status'] == 'PASS'
    assert read(DATA_PARENT / 'verification.json')['status'] == 'PASS'
    hashes = {**model_cfg['code_hashes'], **data_cfg['code_hashes']}
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    paths = [Path(__file__), BASE / 'scripts/slurm_list_ambiguity_transfer.sh',
             BASE / 'scripts/verify_list_ambiguity_transfer.py']
    cfg = dict(model_cfg, task=None, tasks=TASKS, betas=BETAS, seeds=SEEDS,
        variants=VARIANTS, data_parent=str(DATA_PARENT), model_parent=str(MODEL_PARENT),
        data_parent_protocol_sha256=digest(DATA_PARENT / 'protocol.json'),
        model_parent_protocol_sha256=digest(MODEL_PARENT / 'protocol.json'),
        primary_models=['LIST_v1', 'gaussian_learned', 'energy_20'],
        additional_control='point_matched repeats original deterministic encoder with the exact grouped training used by the conditional variants',
        baseline_origin='freshly run archived 8650 residual LIST function; NOT earlier 8638 mean+learned-item-bias LIST',
        training_implementation='byte-identical small_task_comparison.list_run and list_ambiguity_story.train; only per-process TASK/PRIOR/PARENT data paths are overridden',
        evaluation='all48 final models and predictions frozen before test labels; same two beta settings and three seeds as8913; exploratory historical data',
        adaptation='no new architecture, objective, epochs, step-size, chains, optimizer or batch-size tuning on these two datasets',
        code_hashes={**hashes, **{str(p.resolve()): digest(p) for p in paths}})
    save(root / 'protocol.json', cfg)
    source_records = {}
    for task, overlap in TASKS.items():
        sub = root / task
        sub.mkdir()
        save(sub / 'protocol.json', dict(cfg, task=task))
        for seed in SEEDS:
            query = DATA_PARENT / task / f'seed_{seed}/queries/DNN'
            record = read(query / 'complete.json')
            assert record['status'] == 'PASS' and record['k'] == 100 and record['users'] == overlap
            assert digest(query / 'topk_ids.npz') == record['item_ids_sha256']
            api = np.load(query / 'topk_ids.npz')
            np.testing.assert_array_equal(api['user_ids'], np.arange(overlap))
            assert api['item_ids'].shape == (overlap, 100)
            source_records[f'{task}/{seed}'] = dict(path=str(query / 'topk_ids.npz'), sha256=record['item_ids_sha256'])
        for beta in BETAS:
            old = DATA_PARENT / task / f'beta_{beta}'
            cell = sub / f'beta_{beta}'
            cell.mkdir()
            manifest = read(old / 'manifest.json')
            assert manifest['overlap_users'] == overlap and manifest['cross_domain_raw_item_overlap'] == 0
            for path, expected in manifest['hashes'].items():
                if path != 'data/test.csv':
                    assert digest(old / path) == expected, path
            for name in ['data', 'baseline', 'inner_baseline']:
                (cell / name).symlink_to((old / name).resolve(), target_is_directory=True)
            save(cell / 'manifest.json', manifest)
            for variant in VARIANTS:
                for seed in SEEDS:
                    output(root, task, variant, beta, seed).mkdir(parents=True)
    # Baselines precede conditional tasks, whose common-initialization checks
    # depend on the corresponding fresh baseline completion record.
    tasks = [dict(task=task, variant=variant, beta=beta, seed=seed)
             for variant in VARIANTS for task in TASKS for beta in BETAS for seed in SEEDS]
    save(root / 'tasks.json', tasks)
    save(root / 'source_records.json', source_records)
    save(root / 'data_verification.json', dict(status='PASS', no_test_labels_opened=True,
        same_source_id_lists=True, unchanged_model_code=True, unchanged_parameters_from8913=True,
        disjoint_catalogs=True, model_count=len(tasks)))
    print('TRANSFER_PREPARED', root, 'models', len(tasks), flush=True)


def train(root, task, variant, beta, seed):
    check_code(root)
    sub = root / task
    if variant == 'LIST_v1':
        archived.PRIOR = DATA_PARENT
        archived.TASK = task
        archived.list_run(sub, seed, beta, 'real')
    else:
        conditional.PRIOR = DATA_PARENT
        conditional.TASK = task
        conditional.PARENT = sub
        conditional.train(sub, variant, beta, seed)


def worker(root, index):
    tasks = read(root / 'tasks.json')
    while True:
        waiting = False
        claimed = False
        for task in tasks:
            out = output(root, **task)
            if (out / 'claim.json').exists():
                continue
            if task['variant'] != 'LIST_v1':
                base = output(root, task['task'], 'LIST_v1', task['beta'], task['seed'])
                if not (base / 'complete.json').exists():
                    waiting = True
                    continue
            try:
                with (out / 'claim.json').open('x') as handle:
                    json.dump(dict(worker=index, gpu=os.environ['CUDA_VISIBLE_DEVICES']), handle)
            except FileExistsError:
                continue
            print('TRANSFER_TASK', index, task, flush=True)
            command = [sys.executable, '-u', str(Path(__file__)), 'train', '--root', str(root)]
            for key, value in task.items():
                command += ['--' + key, str(value)]
            with (out / 'train.log').open('w') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
            claimed = True
        if not waiting:
            break
        if not claimed:
            time.sleep(1)


def evaluate(root):
    check_code(root)
    assert read(root / 'selftest_cuda.json')['status'] == 'PASS'
    tasks = read(root / 'tasks.json')
    frozen = {}
    for task in tasks:
        out = output(root, **task)
        done = read(out / 'complete.json')
        assert done['status'] == 'PASS' and done['test_labels_opened'] is False
        for name, key in [('final.pt', 'final_sha256'), ('test.npy', 'test_sha256'), ('selection.json', 'selection_sha256')]:
            assert digest(out / name) == done[key]
        frozen[str(out.relative_to(root))] = digest(out / 'complete.json')
    assert len(frozen) == 48
    with (root / 'evaluation_started.json').open('x') as handle:
        json.dump(frozen, handle, indent=2)
    rows, individual = [], []
    for task in TASKS:
        for beta in BETAS:
            cell = root / task / f'beta_{beta}'
            manifest = read(cell / 'manifest.json')
            assert digest(cell / 'data/test.csv') == manifest['hashes']['data/test.csv']
            data = frame(cell / 'data/test.csv')
            np.testing.assert_array_equal(data[:, :2], np.load(cell / 'data/test_features.npy'))
            y = data[:, 2]
            mean = predict(dict(np.load(cell / 'baseline/statistics.npz')), data[:, 1].astype(int), 'item_mean')
            controls = {variant: {seed: conditional.metric(np.load(output(root, task, variant, beta, seed) / 'test.npy'), y)
                                  for seed in SEEDS} for variant in ['LIST_v1', 'point_matched']}
            for variant in ['item_mean', *VARIANTS]:
                values = []
                for seed in SEEDS:
                    if variant == 'item_mean':
                        metrics, selection = conditional.metric(mean, y), dict(epoch=0, lambda_value=0.)
                    else:
                        out = output(root, task, variant, beta, seed)
                        metrics = conditional.metric(np.load(out / 'test.npy'), y)
                        selection = read(out / 'selection.json')
                    row = dict(task=task, beta=beta, variant=variant, seed=seed,
                               epoch=selection['epoch'], lambda_value=selection['lambda_value'], **metrics)
                    for name in ['mae', 'rmse']:
                        for reference in ['LIST_v1', 'point_matched']:
                            row[f'delta_{name}_vs_{reference}'] = metrics[name] - controls[reference][seed][name]
                    values.append(row)
                    individual.append(row)
                aggregate = dict(task=task, beta=beta, variant=variant)
                for name in ['mae', 'rmse']:
                    aggregate[name] = float(np.mean([v[name] for v in values]))
                    aggregate[name + '_std'] = float(np.std([v[name] for v in values], ddof=1))
                    for reference in ['LIST_v1', 'point_matched']:
                        key = f'delta_{name}_vs_{reference}'
                        aggregate[key] = float(np.mean([v[key] for v in values]))
                rows.append(aggregate)
    for name, data in [('comparison', rows), ('per_seed', individual)]:
        with (root / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    save(root / 'summary.json', dict(status='PASS', comparison=rows, per_seed=individual, protocol=read(root / 'protocol.json')))
    names = {'item_mean': '普通均值', 'LIST_v1': '原版LIST（残差v1，新跑）',
             'point_matched': 'LIST成组训练对照', 'gaussian_learned': '条件高斯：学习方差',
             'energy_20': '条件能量：20步朗之万'}
    directions = {'book_movie': 'Book→Movie', 'book_music': 'Book→Music'}
    lines = ['# 条件跨域建模：另外两个任务的结果', '',
             '同一冻结的条件高斯/能量实现与参数迁移至Book→Movie、Book→Music；两个β档、三个种子。原版对照重新运行当前归档的均值残差LIST代码。', '',
             '| 任务 | β | 模型 | MAE ± SD | RMSE ± SD | ΔMAE 对原版 | ΔRMSE 对原版 |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {directions[row['task']]} | {row['beta']}% | {names[row['variant']]} | {row['mae']:.6f} ± {row['mae_std']:.6f} | {row['rmse']:.6f} ± {row['rmse_std']:.6f} | {row['delta_mae_vs_LIST_v1']:+.6f} | {row['delta_rmse_vs_LIST_v1']:+.6f} |")
    lines += ['', '每项为三个种子单独计算指标后的平均值；SD为样本标准差，不是预测集成。差值为新版本减原版，负值更好。',
              '源DNN Top100、训练数据普通物品均值、排名池化和目标内积相同。条件版沿用8913的64用户成组训练、10轮、方差初始std0.3、范围0.05..2、MALA步长0.25、训练32链/预测128链/20步；不在新任务调参。',
              '原版按2048交互训练，成组版每轮更新次数不同；完整保留成组确定性对照以说明这项限制。两种概率模型只在训练后验中使用目标训练评分，冷用户预测只根据源列表。',
              '验证集选择epoch0..10与全局λ后，从头在完整外层训练集重训。全部48个最终模型和预测冻结后才读取测试标签。复用历史数据，属于探索性结果。', '',
              '[CSV](comparison.csv) · [逐种子](per_seed.csv) · [协议](protocol.json) · [核验](verification.json)']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print('TRANSFER_RESULTS', json.dumps(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'selftest', 'train', 'worker', 'evaluate'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--beta', type=int, choices=BETAS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'train':
        train(root, args.task, args.variant, args.beta, args.seed)
    elif args.stage == 'worker':
        worker(root, args.worker)
    elif args.stage == 'selftest':
        conditional.selftest(root, args.device)
    else:
        globals()[args.stage](root)
