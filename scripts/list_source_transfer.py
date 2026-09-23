"""Replicate DNN-vs-MF source-list effects on the two other cleaned tasks."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from book_movie_protocol import BASE, digest, save, predict
from residual_experiment import seed_all, guard_test_labels
from residual_summary import metrics
from list_target_heads import paired_seed_mean
import list_source_backbones as student

PRIOR = BASE/'runs/residual_study_20260922'
TASKS = ['movie_book', 'movie_music_clean']
NAMES = ['MF', 'DNN']
SEEDS = [2020, 2021, 2022]
read = student.read
# Reuse the already checked LIST training and verification without changing
# historical files. These functions iterate this explicit two-arm list.
student.NAMES = NAMES


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), BASE/'scripts/slurm_list_source_transfer.sh',
             BASE/'scripts/list_source_backbones.py', BASE/'scripts/list_target_heads.py',
             BASE/'scripts/pfcdr_backbone_study.py', BASE/'scripts/book_movie_protocol.py',
             BASE/'scripts/residual_experiment.py', BASE/'scripts/residual_summary.py',
             BASE/'scripts/run_list_encoder.py']
    paths += [BASE/'external/PFCDR'/p for p in ['models.py', 'float_embedding.py', 'inversion.py', 'inversion_config.json']]
    cfg = dict(tasks=TASKS, backbones=NAMES, seeds=SEEDS, seed=2020,
               source_dim=10, source_epochs=10, source_lr=.01, source_weight_decay=0., source_batch=512,
               source_initialization='same source user/item initialization across MF/DNN within seed; author default DNN affine head',
               source_selection='fixed final pretrain epoch 10; no target labels or source-validation search',
               k=100, seen_item_filter=False, score_clipping=False, learner_uses_scores=False,
               dim=32, hidden=64, dropout=.1, epochs=10, lr=.001, weight_decay=.00001, batch_size=2048,
               student='unchanged original LIST with dot product, global train mean and item bias',
               vocabulary='per task/seed union of returned MF and DNN opaque IDs; shared initial student checkpoint',
               selection='minimum validation RMSE over epochs 1..10 separately for each source backbone and seed',
               unknown_target_items='unchanged frozen smoothed-mean fallback for target items unseen during training',
               randomization='three paired pipeline seeds; source AND student retrained; not a 3x3 factorial experiment',
               test_policy='all 12 source/student pipelines complete and verified before any target test labels',
               scope='two other tasks on existing fixed splits; historical tests; no hyperparameter search or shuffled-list control',
               bootstrap_repeats=2000, bootstrap_seed=994,
               code_hashes={str(p.resolve()):digest(p) for p in paths})
    save(root/'protocol.json', cfg)
    checks = []
    for task in TASKS:
        old = PRIOR/task
        manifest = read(old/'manifest.json')
        for p, h in {**manifest['hashes'], **manifest['baseline_hashes']}.items():
            if p != 'data/test.csv': assert digest(old/p) == h, p
        maps = {d:pd.read_csv(old/f'data/id_maps/{d}_items.csv', dtype={'raw_id':str}) for d in ['source', 'target']}
        intersection = set(maps['source'].raw_id)&set(maps['target'].raw_id)
        assert not intersection
        removed = read(old/'removed_source_items.json')
        assert not set(removed)&set(maps['source'].raw_id)
        assert len(removed) == manifest['cleanup']['removed_items']
        assert manifest['cross_domain_same_user_item_rating_leaks'] == 0
        if task == 'movie_music_clean':
            assert len(removed) == 2441 and manifest['cleanup']['removed_source_ratings'] == 35333
        task_root = root/task
        task_root.mkdir()
        save(task_root/'manifest.json', manifest)
        for seed in SEEDS:
            sub = task_root/f'seed_{seed}'
            sub.mkdir()
            for d in ['data', 'baseline']:
                (sub/d).symlink_to((old/d).resolve(), target_is_directory=True)
            for d in ['queries', 'students']:
                (sub/d).mkdir()
                for name in NAMES: (sub/d/name).mkdir()
            (sub/'selftest.json').symlink_to(root/'selftest.json')
            save(sub/'manifest.json', manifest)
            save(sub/'protocol.json', dict(cfg, task=task, seed=seed))
        checks.append(dict(task=task, raw_item_intersection=0, removed_source_items=len(removed),
                           source_data_sha256=manifest['hashes']['data/source.csv'], cleanup=manifest['cleanup']))
    save(root/'data_verification.json', dict(status='PASS', checks=checks))
    print('PREPARED', root, json.dumps(checks), flush=True)


def source(root, task, seed, name):
    from pfcdr_backbone_study import make_model, initialize, effective_users, table
    sub = root/task/f'seed_{seed}'
    guard_test_labels(sub)
    student.check_code(sub)
    cfg, manifest = read(sub/'protocol.json'), read(sub/'manifest.json')
    out = sub/'queries'/name
    assert not (out/'started.json').exists()
    save(out/'started.json', dict(job=os.environ['SLURM_JOB_ID'], gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                                 device=torch.cuda.get_device_name(), torch=torch.__version__, task=task, seed=seed, backbone=name))
    assert digest(sub/'data/source.csv') == manifest['hashes']['data/source.csv']
    frame = pd.read_csv(sub/'data/source.csv', header=None).to_numpy()
    assert len(frame) == manifest['rows']['source']
    seed_all(seed)
    m = make_model(name, manifest, cfg['source_dim'])
    initialization = initialize(m, name, seed)
    save(out/'initialization.json', initialization)
    x = torch.tensor(frame[:, :2], device='cuda', dtype=torch.long)
    y = torch.tensor(frame[:, 2], device='cuda', dtype=torch.float32)
    optimizer = torch.optim.Adam(m.src_model.parameters(), lr=cfg['source_lr'], weight_decay=cfg['source_weight_decay'])
    generator = torch.Generator(device='cuda').manual_seed(seed+4000)
    history = []
    started = time.monotonic()
    for epoch in range(1, cfg['source_epochs']+1):
        tick = time.monotonic()
        m.train()
        total = torch.zeros((), device='cuda')
        for idx in torch.randperm(len(x), generator=generator, device='cuda').split(cfg['source_batch']):
            optimizer.zero_grad(set_to_none=True)
            loss = (m(x[idx], 'train_src')-y[idx]).square().mean()
            loss.backward()
            optimizer.step()
            total += loss.detach()*len(idx)
        row = dict(epoch=epoch, train_mse=total.item()/len(x), seconds=time.monotonic()-tick)
        assert np.isfinite(row['train_mse'])
        history.append(row)
        save(out/'history.json', history)
        print('SOURCE_EPOCH', task, seed, name, json.dumps(row), flush=True)
    # Only source weights are serialized. Random, unused target-domain model
    # tables are never passed to the LIST student.
    torch.save(m.src_model.state_dict(), out/'source.pt')
    del frame, x, y, optimizer
    m.eval()
    ids, scores, checks = [], [], []
    with torch.no_grad():
        u = effective_users(m, name, 'src')[:manifest['overlap_users']]
        v = table(m, name, 'src').iid_embedding.weight
        for start in range(0, len(u), 128):
            values, indices = (u[start:start+128]@v.T).topk(cfg['k'], dim=1, sorted=True)
            ids.append(indices.cpu().numpy().astype(np.int32)); scores.append(values.cpu().numpy())
        ids, scores = np.concatenate(ids), np.concatenate(scores)
        assert ids.shape == scores.shape == (manifest['overlap_users'], cfg['k'])
        assert np.isfinite(scores).all() and (np.diff(scores, axis=1) <= 0).all()
        assert (np.diff(np.sort(ids, axis=1), axis=1) > 0).all()
        for uid in [0, len(u)//3, len(u)//2, len(u)-1]:
            native = []
            for first in range(0, len(v), 65536):
                items = torch.arange(first, min(first+65536, len(v)), device='cuda')
                native.append(m(torch.stack([torch.full_like(items, uid), items], 1), 'train_src'))
            native = torch.cat(native)
            fast = u[uid]@v.T
            torch.testing.assert_close(native, fast, rtol=3e-5, atol=3e-5)
            idx = torch.tensor(ids[uid], dtype=torch.long, device='cuda')
            torch.testing.assert_close(native[idx], torch.tensor(scores[uid], device='cuda'), rtol=3e-5, atol=3e-5)
            kth = native.topk(cfg['k']).values[-1]
            assert native[idx].min() >= kth-3e-5*(1+kth.abs())
            checks.append(dict(user=uid, full_catalog_native=True, max_score_error=float((native-fast).abs().max()),
                               native_top100_overlap=int(np.isin(ids[uid], native.topk(cfg['k']).indices.cpu().numpy()).sum())))
    np.savez_compressed(out/'topk_ids.npz', user_ids=np.arange(len(ids), dtype=np.int32), item_ids=ids)
    np.save(out/'scores.npy', scores)
    save(out/'complete.json', dict(status='PASS', task=task, seed=seed, backbone=name, k=cfg['k'], users=len(ids),
                                   source_sha256=digest(out/'source.pt'), source_data_sha256=manifest['hashes']['data/source.csv'],
                                   initial_source_embedding_sha256=initialization['src'], history_sha256=digest(out/'history.json'),
                                   item_ids_sha256=digest(out/'topk_ids.npz'), scores_sha256=digest(out/'scores.npy'),
                                   unique_returned_items=len(np.unique(ids)), source_items=len(v), native_checks=checks,
                                   seen_item_filter=False, score_clipping=False, target_labels_used=False,
                                   seconds=time.monotonic()-started))
    print('SOURCE_EXPORT_PASS', task, seed, name, flush=True)


def verify(root, task, seed):
    sub = root/task/f'seed_{seed}'
    cfg = read(sub/'protocol.json')
    hashes = set()
    for name in NAMES:
        out = sub/'queries'/name
        c = read(out/'complete.json')
        for p, key in [('source.pt', 'source_sha256'), ('history.json', 'history_sha256'),
                       ('topk_ids.npz', 'item_ids_sha256'), ('scores.npy', 'scores_sha256')]:
            assert digest(out/p) == c[key]
        h = read(out/'history.json')
        assert [v['epoch'] for v in h] == list(range(1, cfg['source_epochs']+1))
        hashes.add(c['initial_source_embedding_sha256'])
        assert c['status'] == 'PASS' and not c['target_labels_used']
    assert len(hashes) == 1
    student.verify(sub)


def evaluate(root):
    student.check_code(root)
    cfg = read(root/'protocol.json')
    assert read(root/'data_verification.json')['status'] == 'PASS'
    assert read(root/'selftest.json')['status'] == 'PASS'
    frozen = {}
    for task in TASKS:
        for seed in SEEDS:
            sub = root/task/f'seed_{seed}'
            assert read(sub/'verification.json')['status'] == 'PASS'
            for name in NAMES:
                out = sub/'students'/name
                c = read(out/'complete.json')
                assert c['status'] == 'PASS' and digest(out/'selection.json') == c['selection_sha256']
                frozen[f'{task}/{seed}/{name}'] = c['selection_sha256']
    with (root/'evaluation_started.json').open('x') as f: json.dump(frozen, f, indent=2)
    tasks, rows, aggregate_rows = {}, [], []
    for task in TASKS:
        task_root = root/task
        ref = task_root/'seed_2020'
        manifest = read(task_root/'manifest.json')
        assert digest(ref/'data/test.csv') == manifest['hashes']['data/test.csv']
        test = pd.read_csv(ref/'data/test.csv', header=None).to_numpy()
        np.testing.assert_array_equal(test[:, :2], np.load(ref/'data/test_features.npy'))
        y = test[:, 2]
        _, users = np.unique(test[:, 0], return_inverse=True)
        stats = dict(np.load(ref/'baseline/statistics.npz'))
        baselines = {n:predict(stats, test[:, 1].astype(int), n) for n in ['item_mean', 'smoothed_mean_5', 'item_median']}
        predictions, models = {}, {}
        for name in NAMES:
            predictions[name] = np.stack([np.load(task_root/f'seed_{s}/students'/name/'test.npy') for s in SEEDS])
            per_seed = {}
            for k, seed in enumerate(SEEDS):
                sub = task_root/f'seed_{seed}'
                out = sub/'students'/name
                selection = read(out/'selection.json')
                assert digest(out/'test.npy') == selection['test_sha256']
                assert selection['list_sha256'] == digest(sub/'queries'/name/'topk_ids.npz')
                m = metrics(predictions[name][k], y, users)
                per_seed[str(seed)] = dict(metrics=m, selection=selection)
                rows.append(dict(task=task, source_backbone=name, seed=seed, epoch=selection['epoch'],
                                 validation_rmse=selection['validation_rmse'], mae=m['mae'], rmse=m['rmse']))
            agg = {key:dict(mean=float(np.mean([v['metrics'][key] for v in per_seed.values()])),
                           std=float(np.std([v['metrics'][key] for v in per_seed.values()], ddof=1)))
                   for key in ['mae', 'rmse', 'user_macro_mae']}
            models[name] = dict(per_seed=per_seed, aggregate=agg)
            aggregate_rows.append(dict(task=task, source_backbone=name, mae_mean=agg['mae']['mean'], mae_std=agg['mae']['std'],
                                       rmse_mean=agg['rmse']['mean'], rmse_std=agg['rmse']['std']))
        comparisons = {n:paired_seed_mean(predictions['DNN'], other, y, users, cfg)
                       for n, other in {**{n:p[None, :] for n, p in baselines.items()}, 'LIST_MF':predictions['MF']}.items()}
        for c in comparisons.values():
            c['interpretation'] = 'negative favors DNN-source LIST; arithmetic mean of per-pipeline metrics; user-cluster CI conditional on three paired source/student seeds'
        result = dict(task=task, cleanup=manifest['cleanup'], baselines={n:metrics(p, y, users) for n, p in baselines.items()},
                      models=models, DNN_comparisons=comparisons,
                      list_comparisons={str(seed):read(task_root/f'seed_{seed}/shared.json')['list_comparisons'] for seed in SEEDS})
        tasks[task] = result
        save(task_root/'summary.json', dict(status='PASS', **result))
    save(root/'summary.json', dict(status='PASS', protocol=cfg, tasks=tasks))
    for filename, values in [('per_seed.csv', rows), ('comparison.csv', aggregate_rows)]:
        with (root/filename).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
    lines = ['# DNN 源列表在另外两个任务上的复核', '',
             '每任务三组配对种子；MF/DNN 源模型和 LIST 学生都重新训练；均值 ± 样本标准差，不是预测集成。', '',
             '| 任务 | 方法 | MAE ↓ | RMSE ↓ |', '| --- | --- | ---: | ---: |']
    for task, t in tasks.items():
        for name, m in t['baselines'].items(): lines.append(f"| {task} | {name} | {m['mae']:.6f} | {m['rmse']:.6f} |")
        for r in [v for v in aggregate_rows if v['task'] == task]:
            lines.append(f"| {task} | LIST，源 {r['source_backbone']} | {r['mae_mean']:.6f} ± {r['mae_std']:.6f} | {r['rmse_mean']:.6f} ± {r['rmse_std']:.6f} |")
    lines += ['', '所有 LIST 使用原内积评分，相同初始参数、批次顺序和训练设置，仅源 Top-100 不同。',
              '源/学生均 10 epoch；源固定最后一轮，学生仅以验证 RMSE 选 epoch。全部 12 次训练冻结核验后统一读测试标签。',
              'Movie→Music 沿用清洗数据：源域删除 2441 个共享物品、35333 条评分；再次核对源目标目录交集为零。',
              '本轮不调参、不改变 K、不新增打乱列表对照；数据为历史随机用户划分，无时间过滤。',
              '三个种子同时改变源训练和学生训练，并非两类随机性的全因子分解。',
              '配对用户簇区间条件于已训练的三组模型，没有多重检验校正；不能直接与 Book→Movie 单种子结果视作相同重复量。',
              '细节见 summary.json、per_seed.csv、data_verification.json、各 seed 的 verification.json。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('LIST_TRANSFER_PASS', json.dumps(aggregate_rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'selftest', 'source', 'shared', 'train', 'verify', 'evaluate'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', choices=TASKS)
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--backbone', choices=NAMES)
    a = p.parse_args()
    root = a.root.resolve()
    if a.stage in ['source', 'shared', 'train', 'verify']:
        assert a.task is not None and a.seed is not None
        sub = root/a.task/f'seed_{a.seed}'
    if a.stage == 'source': source(root, a.task, a.seed, a.backbone)
    elif a.stage == 'train': student.train(sub, a.backbone)
    elif a.stage == 'shared': student.shared(sub)
    elif a.stage == 'verify': verify(root, a.task, a.seed)
    elif a.stage == 'selftest': student.selftest(root)
    else: globals()[a.stage](root)
