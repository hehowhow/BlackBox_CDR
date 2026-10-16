"""Trusted coarse-list release x deterministic/Gaussian transfer, fixed 2x2.

The publisher and privacy evaluator are separate trusted processes. Target
training consumes either a released raw-token control or coarse-token list,
never both. Filesystem audit hooks enforce the declared experimental boundary;
this is an executable protocol simulation, not operating-system isolation.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from book_movie_protocol import BASE, digest, predict, save
from list_source_backbones import read, check_code, student_guard
from residual_experiment import seed_all
from small_task_comparison import ResidualList, frame
from list_ambiguity_story import GroupedRatings, correction, metric, parameter_hash
from list_sid_story import old_baseline

DATA_PARENT = BASE / 'runs/paper_list_dnn_8638'
TASKS = {'music_movie_clean': 17904, 'book_movie': 37388, 'book_music': 16738}
BETAS, SEEDS = [20, 80], [2020, 2021, 2022]
VARIANTS = ['raw_point', 'raw_gaussian', 'coarse_point', 'coarse_gaussian']
TOKENIZER_SEED = 20260927


def output(root, task, variant, beta, seed):
    return root / 'models' / task / f'beta_{beta}' / f'seed_{seed}' / variant


def release_dir(root, task, beta, seed, phase):
    return root / 'releases' / task / f'beta_{beta}' / f'seed_{seed}' / phase


def trusted_dir(root, task, beta, seed, phase):
    return root / 'trusted' / task / f'beta_{beta}' / f'seed_{seed}' / phase


def hash_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    cfg = read(BASE / 'runs/small_task_comparison_8650/protocol.json')
    for path, expected in cfg['code_hashes'].items():
        assert digest(Path(path)) == expected, path
    additions = ['list_coarse_sid_study.py', 'list_coarse_sid_release.py',
                 'selftest_list_coarse_sid_release.py', 'list_coarse_sid_model.py',
                 'selftest_list_coarse_sid_model.py', 'list_coarse_sid_privacy.py',
                 'slurm_list_coarse_sid.sh', 'list_ambiguity_story.py',
                 'list_ambiguity_model.py', 'list_sid_story.py']
    cfg.update(tasks=TASKS, task=None, seeds=SEEDS, betas=BETAS, variants=VARIANTS,
        user_batch_size=64, eval_user_batch=128, latent_std=.5,
        logvar_bounds=[math.log(.05 ** 2), math.log(1. ** 2)],
        observation_variance=1., energy_bound=5., train_chains=32, eval_chains=128,
        mala_step=.25, tokenizer_seed=TOKENIZER_SEED, code_capacity=256,
        code_minimum_training_items=8,
        study='fixed 2x2 raw/coarse list x point/conditional Gaussian; no tuning',
        release='trusted training-list cooccurrence SVD32+single-level256codes; tiny groups merged; unknown IDs to largest training group; preserves all100 ranks and code repetitions',
        boundary='coarse student never opens raw queries, raw-token release, raw control models or trusted mapping; code learning uses no target labels/history/teacher weights',
        privacy='exact source recommendation item association; deterministic coarse disclosure, NOT DP, user anonymity or sensitive-interest hiding',
        training='all four arms64complete overlap users/update, same shuffle and updates/epoch; target-only ratings only fixed means; same10epochs/Adam/lambdas',
        initialization='same seed raw-vocabulary reference provides all non-source parameters; coarse table fixed capacity256 across phases; point/Gaussian same initialized parameters',
        selection='inner-train-only tokenizer and model, validation epoch/lambda; fresh outer-train-only tokenizer and refit; test-label-free until all72predictions frozen',
        baseline='historical original LISTv1 2048interaction batches separately reported, exact frozen predictions; NOT the grouped point arm',
        gaussian='fixed initstd.5,bounds.05..1,observation variance1; exact joint marginal likelihood; analytic conditional-mean inference',
        prior_sid_difference='not8993target-supervised three parallel codebooks: new frozen trusted-side cooccurrence coarse release',
        privacy_evaluation='trusted evaluator only after model freeze; train-frequency and code/rank attackers on heldout source outputs; no feedback to tuning',
        evaluation='historical exploratory splits; three separate seed metrics, not prediction ensemble',
        slurm_job=os.environ.get('SLURM_JOB_ID'))
    cfg['code_hashes'].update({str(BASE / 'scripts' / p): digest(BASE / 'scripts' / p) for p in additions})
    save(root / 'protocol.json', cfg)
    references = {}
    for task, overlap in TASKS.items():
        for beta in BETAS:
            old = DATA_PARENT / task / f'beta_{beta}'
            cell = root / 'data' / task / f'beta_{beta}'
            cell.mkdir(parents=True)
            manifest = read(old / 'manifest.json')
            assert manifest['overlap_users'] == overlap and manifest['cross_domain_raw_item_overlap'] == 0
            for path, expected in manifest['hashes'].items():
                if path != 'data/test.csv':
                    assert digest(old / path) == expected, path
            for name in ['data', 'baseline', 'inner_baseline']:
                (cell / name).symlink_to((old / name).resolve(), target_is_directory=True)
            save(cell / 'manifest.json', manifest)
            for seed in SEEDS:
                ref = old_baseline(task, beta, seed)
                done = read(ref / 'complete.json')
                assert digest(ref / 'test.npy') == done['test_sha256']
                references[f'{task}/{beta}/{seed}'] = dict(path=str(ref), test_sha256=done['test_sha256'])
                for variant in VARIANTS:
                    output(root, task, variant, beta, seed).mkdir(parents=True)
    tasks = [dict(task=t, beta=b, seed=s, variant=v)
             for s in SEEDS for t in TASKS for b in BETAS for v in VARIANTS]
    save(root / 'tasks.json', tasks)
    save(root / 'baseline_references.json', references)
    save(root / 'data_verification.json', dict(status='PASS', model_count=len(tasks),
        no_test_labels_opened=True, all_training_inputs_hashed=True, deduplicated=True))
    print('COARSE_STUDY_PREPARED', root, len(tasks), flush=True)


def publish(root, task, beta, seed):
    """Trusted process; does not read target rating values or test labels."""
    from list_coarse_sid_release import fit_release
    check_code(root)
    cfg = read(root / 'protocol.json')
    cell = root / 'data' / task / f'beta_{beta}'
    query = DATA_PARENT / task / f'seed_{seed}/queries/DNN'
    record = read(query / 'complete.json')
    assert digest(query / 'topk_ids.npz') == record['item_ids_sha256']
    api = np.load(query / 'topk_ids.npz')
    raw_ids = api['item_ids']
    np.testing.assert_array_equal(api['user_ids'], np.arange(TASKS[task]))
    assert raw_ids.shape == (TASKS[task], 100)
    raw_vocabulary = np.unique(raw_ids)
    raw_tokens = np.searchsorted(raw_vocabulary, raw_ids).astype(np.int64) + 1
    for phase in ['select', 'refit']:
        out = release_dir(root, task, beta, seed, phase)
        trust = trusted_dir(root, task, beta, seed, phase)
        out.mkdir(parents=True)
        trust.mkdir(parents=True)
        name = 'inner_train_users.npy' if phase == 'select' else 'train_users.npy'
        users = np.load(cell / 'data' / name).astype(np.int64)
        users = users[users < TASKS[task]]
        assert not np.intersect1d(users, np.load(cell / 'data/test_users.npy')).size
        if phase == 'select':
            assert not np.intersect1d(users, np.load(cell / 'data/validation_users.npy')).size
        tick = time.monotonic()
        codes, artifacts, audit = fit_release(raw_ids, users, TOKENIZER_SEED)
        assert codes.shape == raw_ids.shape and codes.min() >= 1 and codes.max() <= cfg['code_capacity']
        np.savez_compressed(trust / 'mapping.npz', **artifacts)
        np.save(trust / 'fit_users.npy', users)
        save(trust / 'audit.json', audit)
        np.savez_compressed(out / 'coarse.npz', user_ids=api['user_ids'], item_ids=codes)
        np.savez_compressed(out / 'raw.npz', user_ids=api['user_ids'], item_ids=raw_tokens)
        save(out / 'release.json', dict(status='PASS', phase=phase,
            fit_user_count=len(users), fit_users_sha256=hash_array(users),
            raw_vocab_size=len(raw_vocabulary), coarse_vocab_size=cfg['code_capacity'],
            active_codes=int(len(np.unique(codes))),
            coarse_sha256=digest(out / 'coarse.npz'), raw_sha256=digest(out / 'raw.npz'),
            no_target_labels_read=True, no_heldout_users_fit=True,
            source_query_sha256=record['item_ids_sha256'], tokenizer_seed=TOKENIZER_SEED))
        print('RELEASE_READY', task, beta, seed, phase, f'{time.monotonic()-tick:.1f}s', flush=True)


def target_guard(root, variant, out):
    student_guard(root)
    trusted = (root / 'trusted').resolve()
    opened = set()
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        blocked = trusted in path.parents or 'queries' in path.parts or path.name == 'mapping.npz'
        if variant.startswith('coarse'):
            blocked |= path.name == 'raw.npz' or any(p in {'raw_point', 'raw_gaussian'} for p in path.parts)
        else:
            blocked |= path.name == 'coarse.npz'
        if blocked:
            raise RuntimeError(f'Target access outside released interface: {path}')
        if path.suffix in {'.npz', '.npy', '.csv', '.pt'}:
            opened.add(str(path))
    sys.addaudithook(hook)
    probes = [root / 'trusted/probe.npy', DATA_PARENT / 'probe/queries/topk_ids.npz', root / 'test.csv']
    probes.append(root / ('raw.npz' if variant.startswith('coarse') else 'coarse.npz'))
    for path in probes:
        try:
            path.open('rb')
        except RuntimeError:
            continue
        raise AssertionError(f'Guard allowed forbidden access: {path}')
    return opened


def train(root, task, variant, beta, seed):
    from list_coarse_sid_model import make_model
    check_code(root)
    assert os.environ.get('SLURM_JOB_ID'), 'Training must run under Slurm'
    cfg = read(root / 'protocol.json')
    out = output(root, task, variant, beta, seed)
    opened = target_guard(root, variant, out)
    save(out / 'started.json', dict(job=os.environ['SLURM_JOB_ID'], gpu=os.environ['CUDA_VISIBLE_DEVICES'], time=time.time()))
    cell = root / 'data' / task / f'beta_{beta}'
    manifest = read(cell / 'manifest.json')
    validation = frame(cell / 'data/validation.csv')
    vx = torch.as_tensor(validation[:, :2], device='cuda', dtype=torch.long)
    tx = torch.as_tensor(np.load(cell / 'data/test_features.npy'), device='cuda', dtype=torch.long)
    kind, family = variant.split('_')
    selection, initials, phases = None, [], {}
    for phase in ['select', 'refit']:
        release = release_dir(root, task, beta, seed, phase)
        released = read(release / 'release.json')
        path = release / f'{kind}.npz'
        assert digest(path) == released[f'{kind}_sha256']
        tokens = np.load(path)['item_ids']
        vocab_size = released[f'{kind}_vocab_size']
        lists = torch.zeros((manifest['target_users'], cfg['k']), device='cuda', dtype=torch.long)
        lists[:len(tokens)] = torch.as_tensor(tokens, device='cuda')
        data_path = cell / ('data/inner_train.csv' if phase == 'select' else 'data/train.csv')
        assert digest(data_path) == manifest['hashes'][str(data_path.relative_to(cell))]
        data = frame(data_path)
        assert not np.intersect1d(data[:, 0], np.load(cell / 'data/test_users.npy')).size
        if phase == 'select':
            assert not np.intersect1d(data[:, 0], validation[:, 0]).size
        bpath = cell / ('inner_baseline' if phase == 'select' else 'baseline')
        base64 = predict(dict(np.load(bpath / 'statistics.npz')), np.arange(manifest['target_items']), 'item_mean')
        base = torch.as_tensor(base64.astype(np.float32), device='cuda')
        seen = torch.as_tensor(np.load(bpath / 'seen.npy'), device='cuda')
        seed_all(seed)
        reference = ResidualList(lists, released['raw_vocab_size'], manifest['target_items'], base, cfg).cuda()
        common = {n: p.detach().clone() for n, p in reference.named_parameters() if not n.startswith('source_items.')}
        del reference
        seed_all(seed)
        model = make_model(lists, vocab_size, manifest['target_items'], base, cfg, family).cuda()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in common:
                    p.copy_(common[name])
        common_hash = hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in common.values())).hexdigest()
        del common
        initials.append(parameter_hash(model))
        assert len(initials) == 1 or initials[0] == initials[1], 'Refit must use same parameter initialization'
        grouped = GroupedRatings(data, manifest['overlap_users'], 'cuda')
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(seed + 200)
        seed_all(seed + 100)
        if phase == 'select':
            baseval = base64[validation[:, 1].astype(int)]
            selection = dict(epoch=0, lambda_value=0., validation_rmse=float(np.sqrt(np.square(baseval-validation[:, 2]).mean())))
            torch.save(model.state_dict(), out / 'selected.pt')
            np.save(out / 'validation.npy', baseval)
        history = []
        epochs = cfg['epochs'] if phase == 'select' else selection['epoch']
        for epoch in range(1, epochs + 1):
            tick = time.monotonic()
            model.train()
            total, updates = 0., 0
            for batch_id, index in enumerate(torch.randperm(len(grouped.users), device='cuda', generator=generator).split(cfg['user_batch_size'])):
                batch = grouped.batch(index, base)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = model.objective(*batch, seed + epoch * 10000 + batch_id)
                assert torch.isfinite(loss), (task, variant, phase, epoch)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * int(batch[-1].sum())
                updates += 1
            row = dict(epoch=epoch, objective=total/grouped.rows, updates=updates, seconds=time.monotonic()-tick)
            if phase == 'select':
                delta = correction(model, vx, seen, seed)
                row['validation'] = {}
                for lam in cfg['lambdas']:
                    pred = baseval + lam * delta
                    score = float(np.sqrt(np.square(pred-validation[:, 2]).mean()))
                    row['validation'][str(lam)] = score
                    if score < selection['validation_rmse']:
                        selection = dict(epoch=epoch, lambda_value=lam, validation_rmse=score)
                        torch.save(model.state_dict(), out / 'selected.pt')
                        np.save(out / 'validation.npy', pred)
            history.append(row)
            save(out / f'{phase}_history.json', history)
            print('COARSE_EPOCH', task, beta, seed, variant, phase, json.dumps(row), flush=True)
        save(out / f'{phase}_history.json', history)
        phases[phase] = dict(common_initial_sha256=common_hash, initial_sha256=initials[-1],
            release_sha256=released[f'{kind}_sha256'], vocab_size=vocab_size,
            overlap_train_rows=grouped.rows, grouped_users=len(grouped.users),
            full_target_train_rows=len(data), epochs=epochs,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            fit_users_sha256=released['fit_users_sha256'])
        if phase == 'select':
            model.load_state_dict(torch.load(out / 'selected.pt', map_location='cuda', weights_only=True))
            np.testing.assert_allclose(baseval+selection['lambda_value']*correction(model,vx,seen,seed), np.load(out/'validation.npy'), atol=1e-7, rtol=0)
            save(out / 'selection.json', selection)
        else:
            torch.save(model.state_dict(), out / 'final.pt')
            pred = base64[tx[:, 1].cpu().numpy()] + selection['lambda_value']*correction(model,tx,seen,seed)
            np.save(out / 'test.npy', pred)
            model.load_state_dict(torch.load(out / 'final.pt', map_location='cuda', weights_only=True))
            np.testing.assert_array_equal(pred, base64[tx[:, 1].cpu().numpy()]+selection['lambda_value']*correction(model,tx,seen,seed))
        del model, optimizer, grouped, lists, data
    save(out / 'access_audit.json', dict(status='PASS', opened=sorted(opened), enforced_interface=kind,
        trusted_mapping_blocked=True, raw_queries_blocked=True, test_labels_blocked=True))
    save(out / 'complete.json', dict(status='PASS', task=task, variant=variant, beta=beta, seed=seed,
        phases=phases, test_labels_opened=False, restored_predictions=True, refit_reinitialized=True,
        final_sha256=digest(out/'final.pt'), test_sha256=digest(out/'test.npy'),
        selection_sha256=digest(out/'selection.json'), finished=time.time()))
    print('COARSE_FROZEN', task, beta, seed, variant, flush=True)


def gpu_available():
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    raw = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, use = [int(x.strip()) for x in raw.strip().split(',')]
    return memory < 1024 and use < 10, dict(gpu=gpu, memory_mib=memory, utilization=use)


def worker(root, index):
    tasks = read(root/'tasks.json')
    announced = False
    while True:
        unclaimed = [t for t in tasks if not (output(root, **t)/'claim.json').exists()]
        if not unclaimed:
            return
        free, usage = gpu_available()
        if not free:
            if not announced:
                print('WAIT_ALLOCATED_GPU_BUSY', usage, flush=True)
                save(root/f'worker_{index}_waiting.json', usage)
                announced = True
            time.sleep(10)
            continue
        if announced:
            print('ALLOCATED_GPU_AVAILABLE', usage, flush=True)
            announced = False
        for task in unclaimed:
            out = output(root, **task)
            try:
                with (out/'claim.json').open('x') as handle:
                    json.dump(dict(worker=index, gpu=os.environ['CUDA_VISIBLE_DEVICES']), handle)
            except FileExistsError:
                continue
            print('COARSE_TASK', index, task, flush=True)
            cmd = [sys.executable, '-u', str(Path(__file__)), 'train', '--root', str(root)]
            for key, value in task.items():
                cmd += ['--'+key, str(value)]
            with (out/'train.log').open('w') as handle:
                subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
            break


def prepare_releases(root):
    """Two CPU publishers, no CUDA contexts, under the same Slurm allocation."""
    from concurrent.futures import ThreadPoolExecutor
    specs = [(t,b,s) for s in SEEDS for t in TASKS for b in BETAS]
    def launch(spec):
        t,b,s = spec
        path = root / 'publisher_logs' / f'{t}_{b}_{s}.log'
        path.parent.mkdir(exist_ok=True)
        with path.open('w') as handle:
            subprocess.run([sys.executable,'-u',str(Path(__file__)),'publish','--root',str(root),
                '--task',t,'--beta',str(b),'--seed',str(s)], stdout=handle,stderr=subprocess.STDOUT,check=True)
        print('PUBLISH_PAIR_COMPLETE', t,b,s, flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(launch,specs))


def evaluate(root):
    from list_coarse_sid_privacy import evaluate_release
    check_code(root)
    tasks = read(root/'tasks.json')
    frozen = {}
    for task in tasks:
        out = output(root,**task)
        done = read(out/'complete.json')
        assert done['status']=='PASS' and not done['test_labels_opened']
        for name,key in [('final.pt','final_sha256'),('test.npy','test_sha256'),('selection.json','selection_sha256')]:
            assert digest(out/name)==done[key]
        assert read(out/'access_audit.json')['status']=='PASS'
        frozen[str(out.relative_to(root))] = digest(out/'complete.json')
    assert len(frozen)==72
    with (root/'evaluation_started.json').open('x') as handle:
        json.dump(dict(models=frozen, time=time.time()),handle,indent=2)
    references = read(root/'baseline_references.json')
    individuals, summary, privacy = [], [], []
    for task in TASKS:
        for beta in BETAS:
            cell = root/'data'/task/f'beta_{beta}'
            manifest=read(cell/'manifest.json')
            assert digest(cell/'data/test.csv')==manifest['hashes']['data/test.csv']
            test=frame(cell/'data/test.csv')
            np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
            y=test[:,2]
            ordinary=predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean')
            metrics_by_seed={}
            for seed in SEEDS:
                ref=references[f'{task}/{beta}/{seed}']
                assert digest(Path(ref['path'])/'test.npy')==ref['test_sha256']
                predictions={'item_mean':ordinary,'LIST_v1':np.load(Path(ref['path'])/'test.npy')}
                predictions.update({v:np.load(output(root,task,v,beta,seed)/'test.npy') for v in VARIANTS})
                metrics_by_seed[seed]={v:metric(p,y) for v,p in predictions.items()}
                records={v:read(output(root,task,v,beta,seed)/'complete.json') for v in VARIANTS}
                for phase in ['select','refit']:
                    assert len({r['phases'][phase]['common_initial_sha256'] for r in records.values()})==1
                    assert len({r['phases'][phase]['grouped_users'] for r in records.values()})==1
                    for kind in ['raw','coarse']:
                        a,b=[records[f'{kind}_{f}']['phases'][phase] for f in ['point','gaussian']]
                        assert a['initial_sha256']==b['initial_sha256'] and a['release_sha256']==b['release_sha256']
                for v in predictions:
                    sel = (dict(epoch=0,lambda_value=0.) if v=='item_mean' else
                        read(Path(ref['path'])/'selection.json') if v=='LIST_v1' else
                        read(output(root,task,v,beta,seed)/'selection.json'))
                    row=dict(task=task,beta=beta,seed=seed,variant=v,epoch=sel['epoch'],lambda_value=sel['lambda_value'],**metrics_by_seed[seed][v])
                    for m in ['mae','rmse']:
                        for control in ['LIST_v1','raw_point','coarse_point']:
                            row[f'delta_{m}_vs_{control}']=row[m]-metrics_by_seed[seed][control][m]
                    individuals.append(row)
                api=np.load(DATA_PARENT/task/f'seed_{seed}/queries/DNN/topk_ids.npz')
                release=release_dir(root,task,beta,seed,'refit')
                trust=trusted_dir(root,task,beta,seed,'refit')
                result=evaluate_release(api['item_ids'],np.load(release/'coarse.npz')['item_ids'],
                    np.load(trust/'fit_users.npy'),np.load(cell/'data/test_users.npy'))
                privacy.append(dict(task=task,beta=beta,seed=seed,**result))
            for v in ['item_mean','LIST_v1',*VARIANTS]:
                selected=[r for r in individuals if r['task']==task and r['beta']==beta and r['variant']==v]
                row=dict(task=task,beta=beta,variant=v)
                for m in ['mae','rmse']:
                    row[m]=float(np.mean([r[m] for r in selected]))
                    row[m+'_std']=float(np.std([r[m] for r in selected],ddof=1))
                    for control in ['LIST_v1','raw_point','coarse_point']:
                        key=f'delta_{m}_vs_{control}'
                        row[key]=float(np.mean([r[key] for r in selected]))
                row['rmse_better_than_coarse_point_seeds']=sum(r['delta_rmse_vs_coarse_point']<0 for r in selected)
                summary.append(row)
    for name,rows in [('comparison',summary),('per_seed',individuals)]:
        with (root/f'{name}.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    save(root/'privacy.json',privacy)
    save(root/'summary.json',dict(status='PASS',comparison=summary,per_seed=individuals))
    save(root/'verification.json',dict(status='PASS',frozen_models=72,common_initialization=True,
        same_input_within_family_comparisons=True,training_source_interface_audited=True,
        all_test_labels_after_freeze=True,three_seeds=True,privacy_evaluated_after_freeze=True))
    names={'item_mean':'普通均值','LIST_v1':'原版LIST v1（历史参照）','raw_point':'原ID＋成组确定性',
           'raw_gaussian':'原ID＋高斯','coarse_point':'粗码＋成组确定性','coarse_gaussian':'粗码＋高斯'}
    labels={'music_movie_clean':'Music→Movie去重','book_movie':'Book→Movie','book_music':'Book→Music'}
    lines=['# 粗码发布与条件高斯：三数据集实验','',f"Slurm作业：{read(root/'protocol.json')['slurm_job']}；72个新模型，三个种子2020/2021/2022，固定配置，无参数搜索。",'',
        '| 任务 | β | 方法 | MAE ± SD | RMSE ± SD | ΔRMSE 对原版 | ΔRMSE 对同粗码确定性 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in summary:
        lines.append(f"| {labels[r['task']]} | {r['beta']}% | {names[r['variant']]} | {r['mae']:.6f} ± {r['mae_std']:.6f} | {r['rmse']:.6f} ± {r['rmse_std']:.6f} | {r['delta_rmse_vs_LIST_v1']:+.6f} | {r['delta_rmse_vs_coarse_point']:+.6f} |")
    lines += ['', '四个新模型均使用64完整用户的成组批次；原版LIST为2048交互批，单独作为历史参照。普通均值、32维、MLP32→64→32、排名权重、10轮、优化器及验证选epoch/λ规则保持不变。',
        '高斯固定评分噪声方差1、初始std0.5、下限0.05、上限1。粗码在可信侧仅用当前训练用户源Top100共现拟合；固定SVD32、单层256码、小桶至少8训练物品。未见源ID映射至最大训练簇。保留所有100个位置和重复代码，不改成分簇聚合。',
        '本轮粗码与8993目标任务监督的三平行码本不同。训练时严格阻止粗码模型读取原列表和映射；同账号的进程访问审计模拟可信发布，不是物理隔离或密码学保证。',
        '物品候选桶不等于用户匿名；确定性粗码不等于差分隐私。恢复攻击仅评价本轮规定的源推荐物品关联，不评价真实行为历史和敏感属性。两个粗码模型的发布信息相同，其发布层隐私指标也相同。',
        '所有模型及预测冻结后才读取目标测试标签；继续使用历史探索划分，结果不代表独立新测试集。', '',
        '[完整CSV](list_coarse_sid_results.csv) · [逐种子CSV](list_coarse_sid_per_seed_results.csv) · [协议](list_coarse_sid_protocol.md)']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    for source,target in [('report.md','list_coarse_sid_experiment.md'),('comparison.csv','list_coarse_sid_results.csv'),('per_seed.csv','list_coarse_sid_per_seed_results.csv'),('privacy.json','list_coarse_sid_privacy_results.json')]:
        shutil.copyfile(root/source,BASE/'docs'/target)
    print('COARSE_STUDY_EVALUATED',root,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','publish','prepare_releases','train','worker','evaluate'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',choices=TASKS)
    p.add_argument('--beta',type=int)
    p.add_argument('--seed',type=int)
    p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--worker',type=int,default=0)
    a=p.parse_args()
    if a.command=='publish':publish(a.root,a.task,a.beta,a.seed)
    elif a.command=='train':train(a.root,a.task,a.variant,a.beta,a.seed)
    elif a.command=='worker':worker(a.root,a.worker)
    else:globals()[a.command](a.root)


if __name__=='__main__':main()
