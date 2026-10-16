"""Single-seed, three-task rank-aware shared reconstruction + residual LIST.

The archived training loop is reused literally. Only the model factory and its
objective (adding declared masked-list supervision) change. All six predictions
are frozen before opening test labels. Baseline predictions are reused by hash.
"""
import argparse
import csv
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

from book_movie_protocol import BASE, digest, save, predict
from list_source_backbones import read, check_code
from small_task_comparison import frame
from list_ambiguity_story import metric, parameter_hash
from list_raw_fixed_ablation import FIXED
from list_shared_reconstruction_model import (PATTERN_DEFAULTS,
    make_shared_reconstruction_model, inherited_parameter_hash)
from list_public_residual_direct_training import (
    train as direct_train, require_direct_authorization, verify_training_equivalence)
import list_coarse_sid_study as archived

PARENT = BASE / 'runs/list_raw_fixed_ablation_10222'
PREVIOUS = BASE / 'runs/list_public_residual_20261016'
TASKS, BETAS, SEEDS = archived.TASKS, archived.BETAS, [2021]
VARIANT = 'raw_energy'
output = archived.output
LABELS = dict(music_movie_clean='Music→Movie（去重）', book_movie='Book→Movie', book_music='Book→Music')
NEW = 'LIST_shared_reconstruction_langevin'
OLD = 'LIST_langevin'
PREV = 'LIST_frequency_residual_langevin'


def fit_users(cell, phase, overlap):
    name = 'inner_train_users.npy' if phase == 'select' else 'train_users.npy'
    users = np.load(cell/'data'/name).astype(np.int64)
    return users[users < overlap]


def prepare(root):
    from selftest_list_shared_reconstruction import run as selftest
    torch.set_num_threads(4)
    check_code(PARENT)
    assert read(PARENT/'verification.json')['status'] == 'PASS'
    assert read(PREVIOUS/'independent_verification.json')['status'] == 'PASS'
    root.mkdir(parents=True, exist_ok=False)
    cfg = dict(FIXED, task=None, tasks=TASKS, seeds=SEEDS, betas=BETAS,
               variants=[VARIANT], mala_step=.25, reference=str(PARENT),
               previous_frequency_reference=str(PREVIOUS),
               study='Raw LIST+MALA: learned shared-pattern reconstruction and rank-aware residual attention',
               shared_reconstruction=dict(PATTERN_DEFAULTS),
               input='Identical DNN Top100 raw IDs and ranks; no frequency statistics, SID or source features',
               encoder='Four learned slot queries; normalized item+rank keys and log-rank prior; rank-only slot decoder; item-minus-reconstruction residual attention; two output projections',
               reconstruction='Training-user lists only; uniformly mask exactly20/100 ranks; masked IDs excluded from keys/values; full softmax over training-observed IDs only; tied ID classifier',
               auxiliary_objective='Original recommendation objective +0.1*mean_masked_ID_CE/log(training_ID_count)',
               initialization='Inherited parameters and RNG match official raw-ID baseline; zero fusion-attention queries and identity projections reproduce original rank pooling',
               training='Same grouped64-user batches, optimizer,10epochs,validation selection,from-scratch refit; declared masked-ID auxiliary objective is new',
               sampling='Unchanged archived Gaussian base,conditional energy,MALA and target scorer',
               selection='Epoch0..10 and original lambda grid on validation; lambda0 remains allowed',
               tuning='None; seed2021 only; one fixed architecture and auxiliary coefficient',
               interpretation='Added encoder parameters and masked-list supervision; not equal parameter count or identical objective; no guaranteed popularity/personalization disentanglement',
               execution_backend='direct', execution_authorization='direct_authorization.json',
               code_hashes=dict(read(PARENT/'protocol.json')['code_hashes']))
    additions = ['list_shared_reconstruction_model.py', 'list_shared_reconstruction_study.py',
                 'selftest_list_shared_reconstruction.py', 'list_public_residual_direct_training.py',
                 'run_list_shared_reconstruction_direct.sh', 'verify_list_shared_reconstruction.py']
    for name in additions:
        cfg['code_hashes'][str(BASE/'scripts'/name)] = digest(BASE/'scripts'/name)
    save(root/'protocol.json', cfg)
    auth = read(PREVIOUS/'direct_authorization.json')
    auth.update(scope='Continuation: three tasks, betas20/80,seed2021,shared-reconstruction encoder',
                continued_at=time.time())
    save(root/'direct_authorization.json', auth)
    (root/'source_snapshot').mkdir()
    for path in cfg['code_hashes']:
        shutil.copy2(path, root/'source_snapshot'/Path(path).name)
    for name in ['data', 'releases']:
        (root/name).symlink_to((PARENT/name).resolve(), target_is_directory=True)
    tasks = [dict(task=t, beta=b, seed=s, variant=VARIANT) for s in SEEDS for t in TASKS for b in BETAS]
    references, initialization, access = {}, {}, {}
    for request in tasks:
        task, beta, seed = request['task'], request['beta'], request['seed']
        key = f'{task}/{beta}/{seed}'
        cell = root/'data'/task/f'beta_{beta}'
        manifest = read(cell/'manifest.json')
        assert manifest['overlap_users'] == TASKS[task] and manifest['cross_domain_raw_item_overlap'] == 0
        references[key] = {}
        for name, parent in [(OLD, PARENT), (PREV, PREVIOUS)]:
            old = output(parent, **request); done = read(old/'complete.json')
            assert done['status'] == 'PASS'
            for filename, field in [('final.pt','final_sha256'), ('test.npy','test_sha256'), ('selection.json','selection_sha256')]:
                assert digest(old/filename) == done[field]
            references[key][name] = dict(path=str(old), complete_sha256=digest(old/'complete.json'),
                test_sha256=done['test_sha256'], selection_sha256=done['selection_sha256'])
        previous = read(output(PARENT, **request)/'complete.json')
        output(root, **request).mkdir(parents=True)
        for phase in ['select', 'refit']:
            release = archived.release_dir(root, task, beta, seed, phase)
            meta = read(release/'release.json')
            assert digest(release/'raw.npz') == meta['raw_sha256']
            raw = np.load(release/'raw.npz')['item_ids']
            fit = fit_users(cell, phase, TASKS[task])
            assert not np.intersect1d(fit, np.load(cell/'data/test_users.npy')).size
            if phase == 'select':
                assert not np.intersect1d(fit, np.load(cell/'data/validation_users.npy')).size
            assert archived.hash_array(fit) == meta['fit_users_sha256']
            model, record = make_shared_reconstruction_model(torch.from_numpy(raw.copy()),
                meta['raw_vocab_size'], meta['coarse_vocab_size'], manifest['target_items'],
                torch.zeros(manifest['target_items']), cfg, seed, fit)
            assert record['raw_initial_sha256'] == previous['phases'][phase]['initial_sha256']
            model.eval()
            with torch.no_grad():
                users = torch.as_tensor(fit[:3])
                rows = model.lists[users]
                original_h = model.source_items(rows, per_sample_weights=model.rank_weights.expand_as(rows))
                torch.testing.assert_close(model.encode_source(users), original_h, atol=1e-7, rtol=1e-5)
            initialization[f'{key}/{phase}'] = dict(inherited_initial_sha256=record['raw_initial_sha256'],
                new_initial_sha256=parameter_hash(model), extra_parameters=record['extra_trainable_parameters'])
            access[f'{key}/{phase}'] = dict(fit_users=len(fit), fit_users_sha256=meta['fit_users_sha256'],
                raw_sha256=meta['raw_sha256'], candidate_count=record['reconstruction_training_ids'],
                candidate_sha256=archived.hash_array(model.pattern_candidate_ids.cpu().numpy()))
            del model
        assert initialization[f'{key}/select']['new_initial_sha256'] == initialization[f'{key}/refit']['new_initial_sha256']
        print('PREPARED', key, flush=True)
    for filename, value in [('tasks.json', tasks), ('references.json', references),
                            ('initialization.json', initialization), ('training_list_access.json', access)]:
        save(root/filename, value)
    save(root/'selftest_cpu.json', selftest())
    save(root/'training_loop_equivalence.json', verify_training_equivalence())
    names = ['protocol.json','tasks.json','references.json','initialization.json',
             'training_list_access.json','direct_authorization.json','selftest_cpu.json','training_loop_equivalence.json']
    save(root/'plan_frozen.json', dict(time=time.time(), models=len(tasks), files={n:digest(root/n) for n in names}))
    save(root/'preflight.json', dict(status='PASS', models=6, phases=12, original_initial_parameters_match=True,
         initial_condition_matches_rank_pooling=True, train_only_reconstruction_candidates=True, test_labels_opened=False))
    print('SHARED_RECONSTRUCTION_PREPARED', root, flush=True)


def verify_plan(root):
    check_code(root)
    for name, expected in read(root/'plan_frozen.json')['files'].items():
        assert digest(root/name) == expected, name


def train(root, task, variant, beta, seed):
    import list_coarse_sid_model as factory
    verify_plan(root); require_direct_authorization(root); verify_training_equivalence()
    assert variant == VARIANT and seed == 2021 and task in TASKS and beta in BETAS
    key = f'{task}/{beta}/{seed}'
    ref = read(root/'references.json')[key][OLD]
    old = Path(ref['path']); assert digest(old/'complete.json') == ref['complete_sha256']
    previous = read(old/'complete.json')
    initials, access = read(root/'initialization.json'), read(root/'training_list_access.json')
    records, trace = [], []
    original = factory.make_model, archived.parameter_hash, archived.target_guard

    def build(lists, vocab, items, base, cfg, family):
        assert family == 'energy'
        phase = ['select','refit'][len(records)]
        meta = read(archived.release_dir(root, task, beta, seed, phase)/'release.json')
        fit = fit_users(root/'data'/task/f'beta_{beta}', phase, TASKS[task])
        assert archived.hash_array(fit) == access[f'{key}/{phase}']['fit_users_sha256']
        model, record = make_shared_reconstruction_model(lists, vocab, meta['coarse_vocab_size'], items, base, cfg, seed, fit)
        assert inherited_parameter_hash(model) == previous['phases'][phase]['initial_sha256']
        assert parameter_hash(model) == initials[f'{key}/{phase}']['new_initial_sha256']
        assert archived.hash_array(model.pattern_candidate_ids.cpu().numpy()) == access[f'{key}/{phase}']['candidate_sha256']
        records.append(dict(phase=phase, **record))
        native = model.objective
        def objective(*args, **kwargs):
            loss, extra = native(*args, **kwargs)
            row = {name: float(value) for name, value in extra.items()}
            assert all(math.isfinite(v) for v in row.values())
            assert 0 <= row['prior_accept'] <= 1 and 0 <= row['posterior_accept'] <= 1
            trace.append(dict(phase=phase, sample_seed=int(args[-1]), ratings_count=int(args[-2].sum()),
                              total_objective=float(loss.detach()), **row))
            return loss, extra
        model.objective = objective
        return model

    def checked_hash(model, common=False):
        assert not common
        phase = records[-1]['phase']
        assert inherited_parameter_hash(model) == previous['phases'][phase]['initial_sha256']
        actual = parameter_hash(model)
        assert actual == initials[f'{key}/{phase}']['new_initial_sha256']
        return actual

    def guard(study, arm, out):
        def extra(event, args):
            if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(args[0])).resolve()
            if {'trusted','id_maps','tiger_sid_content'}.intersection(path.parts) or path.name == 'sid.npz':
                raise RuntimeError(f'Privileged source access forbidden: {path}')
        sys.addaudithook(extra)
        return original[2](study, arm, out)

    factory.make_model, archived.parameter_hash, archived.target_guard = build, checked_hash, guard
    try:
        direct_train(root, task, variant, beta, seed)
    finally:
        factory.make_model, archived.parameter_hash, archived.target_guard = original
    out = output(root, task, variant, beta, seed); done = read(out/'complete.json')
    for phase in ['select','refit']:
        now, before = done['phases'][phase], previous['phases'][phase]
        for field in ['common_initial_sha256','release_sha256','vocab_size','overlap_train_rows',
                      'grouped_users','full_target_train_rows','fit_users_sha256']:
            assert now[field] == before[field], (phase,field)
        assert now['trainable_parameters'] == before['trainable_parameters'] + initials[f'{key}/{phase}']['extra_parameters']
        history = read(out/f'{phase}_history.json')
        assert len(history) == now['epochs']
        assert all(r['updates'] == math.ceil(now['grouped_users']/64) for r in history)
        assert sum(r['phase']==phase for r in trace) == sum(r['updates'] for r in history)
    save(out/'objective_trace.json', trace)
    save(out/'encoder_audit.json', dict(status='PASS', records=records, original_training_loop=True,
        original_sampler=True, auxiliary_objective_declared=True, train_only_masked_reconstruction=True,
        no_frequency_statistics=True, rank_preserved=True,
        objective_trace_sha256=digest(out/'objective_trace.json')))
    done['encoder_audit_sha256'] = digest(out/'encoder_audit.json'); save(out/'complete.json', done)


def worker(root, index):
    verify_plan(root)
    tasks = read(root/'tasks.json')
    while True:
        if (root/'worker_failure.json').exists():
            raise RuntimeError('Another worker failed')
        pending = [t for t in tasks if not (output(root, **t)/'claim.json').exists()]
        if not pending:
            return
        free, usage = archived.gpu_available()
        if not free:
            save(root/f'worker_{index}_waiting.json', usage); time.sleep(10); continue
        for request in pending:
            out = output(root, **request)
            try:
                with (out/'claim.json').open('x') as handle:
                    json.dump(dict(worker=index,gpu=os.environ['CUDA_VISIBLE_DEVICES']), handle)
            except FileExistsError:
                continue
            command = [sys.executable,'-u',str(Path(__file__)),'train','--root',str(root)]
            for name, value in request.items():
                command += ['--'+name,str(value)]
            print('SHARED_RECONSTRUCTION_START',request,flush=True)
            try:
                with (out/'train.log').open('w') as handle:
                    subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
            except Exception as error:
                save(root/'worker_failure.json',dict(task=request,error=str(error))); raise
            break


def write_csv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def evaluate(root):
    verify_plan(root)
    assert read(root/'selftest_cpu.json')['status'] == 'PASS'
    tasks, refs = read(root/'tasks.json'), read(root/'references.json')
    frozen = {}
    for request in tasks:
        out = output(root,**request); done = read(out/'complete.json')
        assert done['status']=='PASS' and not done['test_labels_opened']
        for file, field in [('test.npy','test_sha256'),('final.pt','final_sha256'),
                            ('selection.json','selection_sha256'),('encoder_audit.json','encoder_audit_sha256')]:
            assert digest(out/file)==done[field]
        assert read(out/'access_audit.json')['status']=='PASS'
        frozen[str(out.relative_to(root))]=digest(out/'complete.json')
    assert len(frozen)==6
    save(root/'evaluation_started.json',dict(time=time.time(),frozen_models=frozen))
    rows, paired = [], []
    for request in tasks:
        task,beta,seed=request['task'],request['beta'],request['seed']
        key=f'{task}/{beta}/{seed}'; cell=root/'data'/task/f'beta_{beta}'
        manifest=read(cell/'manifest.json')
        assert digest(cell/'data/test.csv')==manifest['hashes']['data/test.csv']
        test=frame(cell/'data/test.csv')
        np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
        mean=predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean')
        values={'item_mean':metric(mean,test[:,2])}
        rows.append(dict(task=task,beta=beta,seed=seed,method='item_mean',**values['item_mean'],epoch=0,lambda_value=0.,source_path=str(cell/'baseline')))
        for name,path in [(OLD,Path(refs[key][OLD]['path'])),(PREV,Path(refs[key][PREV]['path'])),(NEW,output(root,**request))]:
            if name!=NEW:
                ref=refs[key][name]
                assert digest(path/'complete.json')==ref['complete_sha256'] and digest(path/'test.npy')==ref['test_sha256']
            pred=np.load(path/'test.npy'); assert pred.shape==(len(test),) and np.isfinite(pred).all()
            selection=read(path/'selection.json'); values[name]=metric(pred,test[:,2])
            if selection['lambda_value']==0:
                np.testing.assert_array_equal(pred,mean)
            rows.append(dict(task=task,beta=beta,seed=seed,method=name,**values[name],epoch=selection['epoch'],
                             lambda_value=selection['lambda_value'],source_path=str(path)))
        paired.append(dict(task=task,beta=beta,seed=seed,
            **{'delta_'+m:values[NEW][m]-values[OLD][m] for m in ['mae','rmse']},
            **{'delta_vs_frequency_'+m:values[NEW][m]-values[PREV][m] for m in ['mae','rmse']}))
    write_csv(root/'results.csv',rows); write_csv(root/'paired.csv',paired)
    for filename in ['results','paired']:
        shutil.copy2(root/f'{filename}.csv',BASE/'docs'/f'list_shared_reconstruction_seed2021_{filename}.csv')
    lines=['LIST＋郎之万：共享模式重构与残差注意力，seed2021',
           '三数据集，β20%/80%，共6组。原ID、DNN Top100；直接使用两张GPU。',
           '新结构：4个共享模式；物品＋排名注意力；仅位置与共享模式参与逐位置重构；残差=原物品向量－重构向量。',
           '训练增加：均匀遮挡20%已返回物品；训练用户列表上的ID交叉熵除以log(训练ID数)，系数0.1。无频次分组或频次加权。',
           '保留原64用户批次、10轮预算、优化器、验证选模规则、条件能量和郎之万参数。新结构增加11584个可训练参数。',
           '所有6组预测冻结后统一评估；单种子结果，不代表多种子稳定性。λ=0表示最终使用普通物品均值。','',
           '数据集\tβ\t方法\tMAE\tRMSE\t所选轮次\tλ']
    translations={'item_mean':'普通物品均值',OLD:'原版LIST＋郎之万',PREV:'上一版频次组残差',NEW:'共享模式重构＋残差'}
    for row in rows:
        lines.append(f"{LABELS[row['task']]}\t{row['beta']}%\t{translations[row['method']]}\t{row['mae']:.6f}\t{row['rmse']:.6f}\t{row['epoch']}\t{row['lambda_value']}")
    lines += ['','相对原版LIST＋郎之万的变化（负数更好）：']
    for row in paired:
        lines.append(f"{LABELS[row['task']]} β{row['beta']}%：ΔMAE={row['delta_mae']:+.6f}，ΔRMSE={row['delta_rmse']:+.6f}")
    (root/'report_zh.txt').write_text('\n'.join(lines)+'\n')
    save(root/'verification.json',dict(status='PASS',models=6,metric_rows=len(rows),all_predictions_frozen_before_test=True,
                                     unchanged_baseline_artifacts=True,seed=2021))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','train','worker','evaluate','check'])
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--task')
    parser.add_argument('--variant',default=VARIANT);parser.add_argument('--beta',type=int)
    parser.add_argument('--seed',type=int,default=2021);parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args()
    if args.action=='train':train(args.root,args.task,args.variant,args.beta,args.seed)
    elif args.action=='worker':worker(args.root,args.index)
    elif args.action=='check':verify_plan(args.root);print('FROZEN_PLAN_PASS',args.root)
    else:globals()[args.action](args.root)
