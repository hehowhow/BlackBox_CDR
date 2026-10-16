"""MF paper-task comparison, seed2020, beta20/50/80, six requested methods.

Reuse the audited PFCDR and LIST trainers. Source queries are newly generated
from the SAME MF weights used by EMCDR/PFCDR. No old DNN predictions are reused.
The only baseline prediction adapter removes the old unseen-item mean fallback
to preserve the author's raw MF forward. Selection uses validation, not test.
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from book_movie_protocol import BASE, digest, save
from list_source_backbones import read, check_code
from residual_experiment import seed_all, guard_test_labels
import small_task_comparison as baseline
import pfcdr_backbone_study as pf
import list_coarse_sid_study as grouped
from list_raw_fixed_ablation import FIXED
from list_raw_fixed_ablation_model import matched_pair
from list_shared_reconstruction_model import PATTERN_DEFAULTS, make_shared_reconstruction_model

DATA = BASE / 'runs/paper_list_dnn_8638'
TASKS = {'book_movie': 37388, 'book_music': 16738, 'music_movie_clean': 17904}
BETAS = [20, 50, 80]
SEED = 2020
METHODS = ['TGTOnly', 'EMCDR', 'PFCDR', 'LIST_V1', 'LIST_V1_Langevin', 'LAPT']


def subroot(root, task):
    return root / 'baselines' / task


def model_output(root, task, beta, method):
    if method in METHODS[:3]:
        return subroot(root, task) / f'beta_{beta}/seed_{SEED}/MF'
    if method == 'LIST_V1':
        return subroot(root, task) / f'beta_{beta}/seed_{SEED}/residual_real'
    return grouped.output(root / method, task, 'raw_energy', beta, SEED)


def prediction_path(root, task, beta, method):
    out = model_output(root, task, beta, method)
    return out / (f'{method}_test.npy' if method in METHODS[:3] else 'test.npy')


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    cfg = read(BASE / 'runs/small_task_comparison_8650/protocol.json')
    cfg.update(FIXED)
    cfg.update(seed=SEED, seeds=[SEED], tasks=TASKS, betas=BETAS,
               methods=METHODS, backbones=['MF'], shared_reconstruction=PATTERN_DEFAULTS,
               batch_size=2048, source_dim=10, source_epochs=10, source_lr=.01,
               source_weight_decay=0., source_batch=512, mala_step=.25,
               pretrain_epochs=10, pretrain_lr=.01, pretrain_weight_decay=0.,
               source_policy='Fresh official MF, same source weights for every method within each task; shared across beta',
               source_batch_policy='Official config:512 for Book tasks,256 for Music-to-Movie',
               target_batch_policy='Official config:512 for Book tasks,256 for Music-to-Movie',
               bridge_batch_policy='Official config:512 for Book tasks,1024 for Music-to-Movie',
               source_interface='Top100 raw item IDs and ranks; no scores, histories, embeddings, SID or coarse codes',
               baseline_prediction='Official raw MF forward, no mean fallback, no rating clipping',
               list_prediction='Ordinary training-item mean plus personalized correction; unseen items use global training mean',
               selection='Independent inner cold-user validation RMSE; fresh full outer refit; no test-based selection',
               paper_differences=['Source-side duplicate-item removal retained as previously authorized',
                   'Sorted raw-ID maps and fixed seed2020 splits; exact author user identities not claimed',
                   'Validation selection and refit instead of best test epoch',
                   'One requested seed instead of the paper ten-run average; fixed released defaults, no grid search'],
               list_training='Original LIST_V1:2048 shuffled interactions, all target training rows; energy/LAPT:64 complete overlap users',
               lapt_change='Unchanged current four-slot ranked reconstruction plus residual attention and masked-ID objective',
               hyperparameter_search=False, primary_k=70000,
               seed_note='Seed2020 with established stage-specific RNG offsets; not byte-identical author process RNG',
               expected_method_results=54)
    names = ['lapt_mf_full.py', 'export_lapt_mf_full.py', 'slurm_lapt_mf_full.sh',
             'list_raw_fixed_ablation_model.py', 'list_shared_reconstruction_model.py',
             'list_coarse_sid_study.py', 'list_coarse_sid_model.py',
             'list_coarse_sid_energy_model.py', 'list_langevin_sensitivity_model.py',
             'list_ambiguity_model.py', 'list_ambiguity_story.py', 'list_raw_fixed_ablation.py',
             'selftest_list_shared_reconstruction.py']
    cfg['code_hashes'].update({str(BASE / 'scripts' / n): digest(BASE / 'scripts' / n) for n in names})
    save(root / 'protocol.json', cfg)
    (root / 'source_snapshot').mkdir()
    for p in cfg['code_hashes']:
        shutil.copy2(p, root / 'source_snapshot' / Path(p).name)
    jobs, counts = [], []
    for task, expected in TASKS.items():
        tr = subroot(root, task)
        tr.mkdir(parents=True)
        m = read(DATA / task / 'manifest.json')
        assert m['overlap_users'] == expected and m['cross_domain_raw_item_overlap'] == 0
        taskcfg = dict(cfg, task=task, source_batch=256 if task == 'music_movie_clean' else 512,
                       target_batch=256 if task == 'music_movie_clean' else 512,
                       bridge_batch=1024 if task == 'music_movie_clean' else 512)
        save(tr / 'protocol.json', taskcfg)
        save(tr / 'manifest.json', m)
        (tr / 'source_data').symlink_to((DATA / task / 'source_data').resolve(), target_is_directory=True)
        assert digest(tr / 'source_data/source.csv') == m['hashes']['data/source.csv']
        (tr / f'sources/seed_{SEED}/MF').mkdir(parents=True)
        (root / 'list_queries' / task / f'seed_{SEED}/queries/MF').mkdir(parents=True)
        jobs.append(dict(id=f'source_{task}', kind='source', task=task, beta=0))
        for beta in BETAS:
            old = DATA / task / f'beta_{beta}'
            cm = read(old / 'manifest.json')
            for p, h in cm['hashes'].items():
                if p != 'data/test.csv':
                    assert digest(old / p) == h, str(old / p)
            for name in ['train', 'inner_train', 'validation', 'test']:
                users = np.load(old / f'data/{name}_users.npy')
                assert len(users) == cm['users'][{'train':'outer_train','inner_train':'inner_train','validation':'validation','test':'test'}[name]]
            assert cm['users']['test'] == round(expected * beta / 100)
            cell = tr / f'beta_{beta}'
            cell.mkdir()
            for name in ['data', 'baseline', 'inner_baseline']:
                (cell / name).symlink_to((old / name).resolve(), target_is_directory=True)
            save(cell / 'manifest.json', cm)
            for arm in ['MF', 'residual_real']:
                (cell / f'seed_{SEED}' / arm).mkdir(parents=True)
            data_alias = root / 'data' / task / f'beta_{beta}'
            data_alias.parent.mkdir(parents=True, exist_ok=True)
            data_alias.symlink_to(cell.resolve(), target_is_directory=True)
            counts.append(dict(task=task, beta=beta, overlap=expected, **cm['users'], rows=cm['rows']))
            jobs.append(dict(id=f'baselines_{task}_{beta}', kind='baselines', task=task, beta=beta))
            for method in METHODS[3:]:
                jobs.append(dict(id=f'{method}_{task}_{beta}', kind=method, task=task, beta=beta))
    for method in METHODS[4:]:
        r = root / method
        r.mkdir()
        save(r / 'protocol.json', dict(cfg, study=method))
        (r / 'data').symlink_to(root / 'data', target_is_directory=True)
        (r / 'releases').symlink_to(root / 'releases', target_is_directory=True)
        for task in TASKS:
            for beta in BETAS:
                model_output(root, task, beta, method).mkdir(parents=True)
    for d in ['claims', 'unit_done', 'logs']:
        (root / d).mkdir()
    save(root / 'tasks.json', jobs)
    save(root / 'dataset_counts.json', counts)
    save(root / 'data_verification.json', dict(status='PASS', cells=9, no_test_labels_opened=True,
         same_data_for_all_methods=True, task3_deduplicated=True))
    save(root / 'plan_frozen.json', dict(time=time.time(), protocol_sha256=digest(root / 'protocol.json'),
         tasks_sha256=digest(root / 'tasks.json'), expected_results=54))
    print('PREPARED', root, len(jobs), 'work units,54 method results', flush=True)


def source_job(root, task):
    tr = subroot(root, task)
    baseline.source(tr, SEED, 'MF')
    src = tr / f'sources/seed_{SEED}/MF'
    cfg, m = read(tr / 'protocol.json'), read(tr / 'manifest.json')
    model = pf.make_model('MF', m, 10)
    model.src_model.load_state_dict(torch.load(src / 'source.pt', map_location='cuda', weights_only=True))
    model.eval()
    with torch.no_grad():
        u = pf.effective_users(model, 'MF', 'src')[:m['overlap_users']]
        v = pf.table(model, 'MF', 'src').iid_embedding.weight
        top = torch.cat([torch.topk(x @ v.T, cfg['k'], dim=1, sorted=True).indices.cpu()
                         for x in u.split(128)]).numpy()
        pairs = torch.tensor([[0, int(top[0,0])], [len(top)-1, int(top[-1,-1])]], device='cuda')
        native = model(pairs, 'train_src').reshape(-1)
        explicit = (u[pairs[:,0]] * v[pairs[:,1]]).sum(1)
        torch.testing.assert_close(native, explicit, rtol=1e-6, atol=1e-6)
    query = root / 'list_queries' / task / f'seed_{SEED}/queries/MF'
    np.savez_compressed(query / 'topk_ids.npz', user_ids=np.arange(len(top)), item_ids=top)
    save(query / 'complete.json', dict(status='PASS', backbone='MF',
         source_sha256=digest(src / 'source.pt'), item_ids_sha256=digest(query / 'topk_ids.npz'),
         native_forward_matches=True, scores_released=False, seen_item_filter=False))
    vocabulary = np.unique(top)
    tokens = np.searchsorted(vocabulary, top).astype(np.int64) + 1
    np.testing.assert_array_equal(vocabulary[tokens - 1], top)
    published = root / 'published' / task
    published.mkdir(parents=True)
    np.savez_compressed(published / 'raw.npz', user_ids=np.arange(len(top)), item_ids=tokens)
    for beta in BETAS:
        for phase in ['select', 'refit']:
            release = grouped.release_dir(root, task, beta, SEED, phase)
            release.mkdir(parents=True)
            (release / 'raw.npz').symlink_to(published / 'raw.npz')
            name = 'inner_train' if phase == 'select' else 'train'
            fit = np.load(tr / f'beta_{beta}/data/{name}_users.npy')
            save(release / 'release.json', dict(raw_vocab_size=len(vocabulary), raw_sha256=digest(release / 'raw.npz'),
                 fit_users_sha256=grouped.hash_array(fit), backbone='MF', source_sha256=digest(src / 'source.pt'),
                 coarse_vocab_size=256, coarse_vocab_note='Archived downstream RNG alignment size only; no coarse codes are constructed or read'))
    print('MF_SOURCE_READY', task, top.shape, len(vocabulary), flush=True)


def baseline_job(root, task, beta):
    # Preserve official MF prediction even on target items absent from training.
    @torch.no_grad()
    def native_em(model, x, seen, base):
        model.eval()
        return torch.cat([model(b, 'test_map').reshape(-1) for b in x.split(4096)]).cpu().numpy()
    original_prediction = pf.rating_predictions
    def native_pf(model, name, vectors, ids, x, fallback=None, seen=None):
        return original_prediction(model, name, vectors, ids, x)
    baseline.em_predict, pf.rating_predictions = native_em, native_pf
    tr = subroot(root, task)
    baseline.baseline_run(tr, SEED, beta, 'MF')
    out = model_output(root, task, beta, 'TGTOnly')
    m = read(tr / 'manifest.json')
    model = pf.make_model('MF', m, 10)
    model.load_state_dict(torch.load(out / 'final.pt', map_location='cuda', weights_only=True))
    model.eval()
    tx = torch.as_tensor(np.load(tr / f'beta_{beta}/data/test_features.npy'), device='cuda', dtype=torch.long)
    with torch.no_grad():
        p = torch.cat([model(x, 'test_tgt').reshape(-1) for x in tx.split(4096)]).cpu().numpy()
        direct = (pf.table(model, 'MF', 'tgt').uid_embedding(tx[:,0]) *
                  pf.table(model, 'MF', 'tgt').iid_embedding(tx[:,1])).sum(1).cpu().numpy()
    np.testing.assert_allclose(p, direct, rtol=1e-6, atol=1e-6)
    assert np.isfinite(p).all()
    np.save(out / 'TGTOnly_test.npy', p)
    record = read(out / 'complete.json')
    record['predictions']['TGTOnly'] = digest(out / 'TGTOnly_test.npy')
    record.update(tgtonly_epochs=10, tgtonly_same_target_checkpoint=True,
                  raw_official_predictions_no_mean_fallback=True)
    save(out / 'complete.json', record)


def v1_job(root, task, beta):
    # The original V1 trainer is unchanged except its source query directory.
    original = inspect.getsource(baseline.list_run)
    assert original.count('queries/DNN') == 1
    adapted = original.replace('queries/DNN', 'queries/MF')
    exec(compile(adapted, str(Path(__file__)) + ':MF_query_adapter', 'exec'), baseline.__dict__)
    baseline.PRIOR, baseline.TASK = root / 'list_queries', task
    baseline.list_run(subroot(root, task), SEED, beta, 'real')
    save(model_output(root, task, beta, 'LIST_V1') / 'trainer_adapter.json',
         dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
              adapted_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
              only_change='queries/DNN -> queries/MF'))


def energy_job(root, task, beta, method):
    import list_coarse_sid_model as factory
    phases = []
    def build(lists, vocab, items, base, cfg, family):
        assert family == 'energy'
        phase = ['select', 'refit'][len(phases)]
        if method == 'LAPT':
            name = 'inner_train' if phase == 'select' else 'train'
            fit = np.load(root / 'data' / task / f'beta_{beta}/data/{name}_users.npy')
            model, record = make_shared_reconstruction_model(lists, vocab, 256, items, base, cfg, SEED, fit)
        else:
            model, record = matched_pair(lists, vocab, 256, items, base, cfg, SEED, 'energy')
        phases.append(dict(phase=phase, **record))
        return model
    factory.make_model = build
    grouped.train(root / method, task, 'raw_energy', beta, SEED)
    save(model_output(root, task, beta, method) / 'initialization_alignment.json', phases)


def run_unit(root, unit):
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm is available; actual allocation required'
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
    torch.set_num_threads(4)
    check_code(root)
    if unit['kind'] == 'source': source_job(root, unit['task'])
    elif unit['kind'] == 'baselines': baseline_job(root, unit['task'], unit['beta'])
    elif unit['kind'] == 'LIST_V1': v1_job(root, unit['task'], unit['beta'])
    else: energy_job(root, unit['task'], unit['beta'], unit['kind'])
    save(root / 'unit_done' / (unit['id'] + '.json'), dict(status='PASS', **unit, finished=time.time()))


def wait_idle_gpu():
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    while True:
        result = subprocess.check_output(['nvidia-smi', '-i', gpu,
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        memory, utilization = map(int, result.split(','))
        processes = subprocess.check_output(['nvidia-smi', '-i', gpu,
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if memory < 1024 and not processes:
            return
        print('WAIT_ALLOCATED_GPU', gpu, result, 'resident_pids', processes, flush=True)
        time.sleep(20)


def worker(root, index):
    jobs = read(root / 'tasks.json')
    while True:
        pending = [j for j in jobs if not (root / 'unit_done' / (j['id'] + '.json')).exists()]
        if not pending: return
        if (root / 'failure.json').exists(): raise RuntimeError('Another work unit failed')
        ready = [j for j in pending if not (root / 'claims' / (j['id'] + '.json')).exists()
                 and (j['kind'] == 'source' or (root / 'unit_done' / ('source_' + j['task'] + '.json')).exists())]
        if not ready:
            time.sleep(10)
            continue
        wait_idle_gpu()
        unit = ready[0]
        claim = root / 'claims' / (unit['id'] + '.json')
        try: handle = claim.open('x')
        except FileExistsError: continue
        with handle:
            json.dump(dict(**unit, worker=index, job=os.environ['SLURM_JOB_ID'],
                           gpu=os.environ['CUDA_VISIBLE_DEVICES'], time=time.time()), handle, indent=2)
        print('START', unit, flush=True)
        with (root / 'logs' / (unit['id'] + '.log')).open('w') as log:
            p = subprocess.run([sys.executable, '-u', str(Path(__file__)), 'unit', '--root', str(root),
                                '--unit-id', unit['id']], stdout=log, stderr=subprocess.STDOUT)
        if p.returncode:
            save(root / 'failure.json', dict(unit=unit, exit_code=p.returncode, time=time.time()))
            raise RuntimeError(f'Failed {unit}')
        print('COMPLETE', unit, flush=True)


def evaluate(root):
    check_code(root)
    jobs = read(root / 'tasks.json')
    assert all((root / 'unit_done' / (j['id'] + '.json')).exists() for j in jobs)
    frozen = {}
    for task in TASKS:
        for beta in BETAS:
            for method in METHODS:
                out = model_output(root, task, beta, method)
                done = read(out / 'complete.json')
                assert done['status'] == 'PASS' and not done['test_labels_opened']
                p = prediction_path(root, task, beta, method)
                expected = done['predictions'][method] if method in METHODS[:3] else done['test_sha256']
                assert digest(p) == expected
                assert digest(out / 'final.pt') == done['final_sha256']
                frozen[f'{task}/{beta}/{method}'] = dict(prediction=str(p), sha256=expected,
                    complete_sha256=digest(out / 'complete.json'), final_sha256=done['final_sha256'])
    assert len(frozen) == 54
    save(root / 'all_predictions_frozen.json', dict(time=time.time(), predictions=frozen))
    rows = []
    for task in TASKS:
        for beta in BETAS:
            cell = root / 'data' / task / f'beta_{beta}'
            m = read(cell / 'manifest.json')
            assert digest(cell / 'data/test.csv') == m['hashes']['data/test.csv']
            data = baseline.frame(cell / 'data/test.csv')
            np.testing.assert_array_equal(data[:,:2], np.load(cell / 'data/test_features.npy'))
            for method in METHODS:
                p = np.load(prediction_path(root, task, beta, method))
                assert p.shape == data[:,2].shape and np.isfinite(p).all()
                diff = p.astype(np.float64) - data[:,2]
                mae, rmse = float(np.abs(diff).mean()), float(np.sqrt(np.square(diff).mean()))
                reference = grouped.metric(p, data[:,2])
                np.testing.assert_allclose([mae,rmse], [reference['mae'],reference['rmse']], atol=1e-12, rtol=0)
                selections = read(model_output(root,task,beta,method) / 'selection.json')
                selection = (dict(epoch=10) if method == 'TGTOnly' else selections[method]) if method in METHODS[:3] else selections
                rows.append(dict(task=task,beta=beta,seed=SEED,method=method,mae=mae,rmse=rmse,
                                 epoch=selection['epoch'],lambda_value=selection.get('lambda_value',''),
                                 test_ratings=len(p),prediction=str(prediction_path(root,task,beta,method))))
    with (root / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(root / 'verification.json', dict(status='PASS', results=54, all_frozen_before_evaluation=True,
         independent_metric_recalculation=True, same_MF_teacher_within_task=True,
         frozen_sha256=digest(root / 'all_predictions_frozen.json'), results_sha256=digest(root / 'results.csv')))
    subprocess.run([sys.executable, str(BASE / 'scripts/export_lapt_mf_full.py'), '--root', str(root)], check=True)
    save(root / 'execution_status.json',dict(status='COMPLETE',results=54,finished=time.time(),
         latex=str(root/'main_results.tex'),pdf=str(root/'main_results.pdf')))
    print('ALL54_COMPLETE',root,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','unit','worker','evaluate'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--index',type=int)
    parser.add_argument('--unit-id')
    args=parser.parse_args()
    if args.command=='unit':
        run_unit(args.root,next(j for j in read(args.root/'tasks.json') if j['id']==args.unit_id))
    elif args.command=='worker':worker(args.root,args.index)
    else:globals()[args.command](args.root)
