"""Extend frozen coarse-SID study9003 with matched conditional energy models.

The production trainer is imported byte-for-byte from9003. Only its model
factory, access guard and initialization verifier are injected in-process.
No old source file, artifact, release or baseline is modified.
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

from book_movie_protocol import BASE, digest, predict, save
from list_source_backbones import check_code, read
from small_task_comparison import frame
from list_ambiguity_story import metric
import list_coarse_sid_study as archived

PARENT = BASE / 'runs/list_coarse_sid_9003'
TASKS, BETAS, SEEDS = archived.TASKS, archived.BETAS, archived.SEEDS
VARIANTS = ['raw_energy', 'coarse_energy']
REFERENCES = ['item_mean', 'LIST_v1', *archived.VARIANTS]
SAMPLING = dict(train_mala_step=.25, eval_mala_step=.25, train_steps=20,
                eval_steps=20, train_chains=32, eval_chains=128, energy_bound=5.)
output = archived.output


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    assert read(PARENT/'verification.json')['status']=='PASS'
    assert read(PARENT/'independent_review.json')['status']=='PASS'
    cfg=read(PARENT/'protocol.json')
    for p,h in cfg['code_hashes'].items():
        assert digest(Path(p))==h,p
    files=['list_coarse_sid_energy_study.py','list_coarse_sid_energy_model.py',
           'selftest_list_coarse_sid_energy.py','slurm_list_coarse_sid_energy.sh',
           'list_langevin_sensitivity_model.py']
    cfg.update(**SAMPLING,variants=VARIANTS,reference_variants=REFERENCES,
        parent=str(PARENT),parent_protocol_sha256=digest(PARENT/'protocol.json'),
        study='36fixed conditional-energy models on same9003raw/coarse releases; no tuning',
        training_implementation='byte-identical9003train with process-local factory/guard/hash-check injection; MALA local generators do not change global dropout state',
        unchanged='stdinit.5,bounds.05..1,noise1,dim32,hidden64,dropout.1;64users/batch;Adam.001/wd1e-5;10epochs;mean+lambda(dot);same epoch/lambda validation rule',
        change='only unfreeze existing64hidden energy network and use conditional EBM maximum-likelihood gradient + MALA conditional-mean inference',
        initialization='all parameter tensors including unused Gaussian energy network match9003same-inputGaussian at both phases; no warm start',
        extra_trainable_parameters=4225,
        sampling='preconditioned Metropolis-adjusted Langevin; independent train/eval controls;20/20steps,.25/.25stepsize,32/128chains,energybound5; no mixing guarantee',
        objective='logged objective is2xEBMMLgradient surrogate, NOT normalized marginal NLL and not comparable in value to Gaussian NLL',
        publication='exactly reuse9003release bytes and training-fold codebooks; no new trusted preprocessing and no changed source access',
        privacy_evaluation='release-level privacy metrics reused unchanged from9003; this study tests recommendation, not new privacy guarantees',
        baseline='9003original Gaussian and point predictions, plus historical LISTv1/ordinary mean; do not mix olderstd.3energy results',
        selection='same inner validation epochs0..10/global lambdas, then fresh outer refit;36new final models frozen before this evaluator opens target test labels',
        evaluation='historical exploratory dataset and test splits already used;3seed averages, no tuning in this run',
        slurm_job=os.environ.get('SLURM_JOB_ID'))
    cfg['code_hashes'].update({str(BASE/'scripts'/p):digest(BASE/'scripts'/p) for p in files})
    save(root/'protocol.json',cfg)
    # Target code can only access the released lists, never the parent's map.
    for name in ['data','releases']:
        (root/name).symlink_to((PARENT/name).resolve(),target_is_directory=True)
    shutil.copyfile(PARENT/'baseline_references.json',root/'baseline_references.json')
    refs={}
    for task in TASKS:
        for beta in BETAS:
            cell=root/'data'/task/f'beta_{beta}'
            m=read(cell/'manifest.json')
            assert m['overlap_users']==TASKS[task] and m['cross_domain_raw_item_overlap']==0
            for p,h in m['hashes'].items():
                if p!='data/test.csv':assert digest(cell/p)==h,p
            for seed in SEEDS:
                for phase in ['select','refit']:
                    rel=archived.release_dir(root,task,beta,seed,phase)
                    record=read(rel/'release.json')
                    for kind in ['raw','coarse']:
                        assert digest(rel/f'{kind}.npz')==record[f'{kind}_sha256']
                for v in archived.VARIANTS:
                    old=output(PARENT,task,v,beta,seed)
                    done=read(old/'complete.json')
                    assert done['status']=='PASS'
                    for name,key in [('test.npy','test_sha256'),('final.pt','final_sha256'),('selection.json','selection_sha256')]:
                        assert digest(old/name)==done[key]
                    refs[f'{task}/{beta}/{seed}/{v}']=dict(path=str(old),
                        complete_sha256=digest(old/'complete.json'),test_sha256=done['test_sha256'])
                for v in VARIANTS:output(root,task,v,beta,seed).mkdir(parents=True)
    tasks=[dict(task=t,beta=b,seed=s,variant=v)
           for s in SEEDS for t in TASKS for b in BETAS for v in VARIANTS]
    save(root/'tasks.json',tasks)
    save(root/'references.json',refs)
    shutil.copyfile(PARENT/'privacy.json',root/'release_privacy_reference.json')
    save(root/'data_verification.json',dict(status='PASS',models=36,
        same_release_files=True,no_codebook_refit=True,no_test_labels_opened=True,
        parent_protocol_sha256=digest(PARENT/'protocol.json')))
    print('ENERGY_EXTENSION_PREPARED',root,flush=True)


def train(root,task,variant,beta,seed):
    from list_coarse_sid_energy_model import make_energy_model
    import list_coarse_sid_model as factory_module
    check_code(root)
    assert variant in VARIANTS
    cfg=read(root/'protocol.json')
    kind=variant.split('_')[0]
    out=output(root,task,variant,beta,seed)
    ref=output(PARENT,task,kind+'_gaussian',beta,seed)
    reference=read(ref/'complete.json')
    original_factory=factory_module.make_model
    original_guard=archived.target_guard
    original_hash=archived.parameter_hash
    hash_checks=[]
    sampled=[]
    model_index=[0]

    def build(lists,vocab,items,base,settings,family):
        assert family=='energy'
        model=make_energy_model(lists,vocab,items,base,settings)
        phase=['select','refit'][model_index[0]]
        model_index[0]+=1
        native=model.objective
        def objective(*args,**kwargs):
            loss,extra=native(*args,**kwargs)
            # Observe the returned diagnostics only; never call the sampler again.
            a,b=float(extra['prior_accept']),float(extra['posterior_accept'])
            assert 0<=a<=1 and 0<=b<=1 and math.isfinite(a+b)
            sampled.append((phase,a,b))
            return loss,extra
        model.objective=objective
        return model

    def guard(study_root,arm,model_out):
        def extra_hook(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0])).resolve()
            forbidden='trusted' in p.parts or 'queries' in p.parts
            if arm.startswith('coarse'):
                forbidden|=bool({'raw_point','raw_gaussian','raw_energy'}.intersection(p.parts)) or p.name=='raw.npz'
            if forbidden:raise RuntimeError(f'Energy student forbidden privileged read: {p}')
        sys.addaudithook(extra_hook)
        # Reject privileged paths before the older hook logs successful opens.
        opened=original_guard(study_root,arm,model_out)
        for p in [PARENT/'trusted/probe.npy',root/'models/raw_energy/probe.pt'] if arm.startswith('coarse') else [PARENT/'trusted/probe.npy']:
            try:p.open('rb')
            except RuntimeError:continue
            raise AssertionError('Guard failed')
        return opened

    def checked_hash(model,common=False):
        actual=original_hash(model,common)
        assert not common
        phase=['select','refit'][len(hash_checks)]
        expected=reference['phases'][phase]['initial_sha256']
        assert actual==expected,(task,beta,seed,variant,phase,'Gaussian initialization mismatch')
        hash_checks.append(dict(phase=phase,sha256=actual,gaussian_sha256=expected))
        return actual

    factory_module.make_model=build
    archived.target_guard=guard
    archived.parameter_hash=checked_hash
    try:
        archived.train(root,task,variant,beta,seed)
    finally:
        factory_module.make_model=original_factory
        archived.target_guard=original_guard
        archived.parameter_hash=original_hash
    done=read(out/'complete.json')
    assert len(hash_checks)==model_index[0]==2
    for phase in ['select','refit']:
        a,b=done['phases'][phase],reference['phases'][phase]
        for key in ['initial_sha256','common_initial_sha256','release_sha256','vocab_size',
                    'overlap_train_rows','grouped_users','full_target_train_rows','fit_users_sha256']:
            assert a[key]==b[key],(phase,key)
        assert a['trainable_parameters']==b['trainable_parameters']+4225
    summaries={}
    for phase in ['select','refit']:
        values=np.asarray([[a,b] for ph,a,b in sampled if ph==phase],dtype=float)
        summaries[phase]=dict(batches=len(values),
            prior_accept_mean=float(values[:,0].mean()) if len(values) else None,
            posterior_accept_mean=float(values[:,1].mean()) if len(values) else None,
            prior_accept_min=float(values[:,0].min()) if len(values) else None,
            posterior_accept_min=float(values[:,1].min()) if len(values) else None)
    save(out/'sampling_audit.json',dict(status='PASS',controls=SAMPLING,
        training=summaries,init_checks=hash_checks,diagnostics_did_not_resample=True,
        inference='source-conditioned prior only, local generator, no observed target ratings'))
    done.update(matched_gaussian_reference=str(ref),full_initialization_matches_gaussian=True,
        extra_trainable_parameters=4225,sampling_audit_sha256=digest(out/'sampling_audit.json'))
    save(out/'complete.json',done)
    print('ENERGY_EXTENSION_FROZEN',task,beta,seed,variant,flush=True)


def worker(root,index):
    tasks=read(root/'tasks.json')
    waiting=False
    while True:
        remaining=[t for t in tasks if not (output(root,**t)/'claim.json').exists()]
        if not remaining:return
        if (root/'worker_failure.json').exists():raise RuntimeError('Another worker failed')
        free,usage=archived.gpu_available()
        if not free:
            if not waiting:
                print('WAIT_ALLOCATED_GPU_BUSY',usage,flush=True)
                save(root/f'worker_{index}_waiting.json',usage)
                waiting=True
            time.sleep(10)
            continue
        waiting=False
        for task in remaining:
            out=output(root,**task)
            try:
                with (out/'claim.json').open('x') as f:json.dump(dict(worker=index,gpu=os.environ['CUDA_VISIBLE_DEVICES']),f)
            except FileExistsError:continue
            command=[sys.executable,'-u',str(Path(__file__)),'train','--root',str(root)]
            for k,v in task.items():command+=['--'+k,str(v)]
            print('ENERGY_EXTENSION_TASK',index,task,flush=True)
            try:
                with (out/'train.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
            except Exception as error:
                save(root/'worker_failure.json',dict(task=task,worker=index,error=str(error)))
                raise
            break


def evaluate(root):
    check_code(root)
    assert read(root/'selftest_cpu.json')['status']=='PASS'
    frozen={}
    for task in read(root/'tasks.json'):
        out=output(root,**task);done=read(out/'complete.json')
        assert done['status']=='PASS' and not done['test_labels_opened']
        assert done['full_initialization_matches_gaussian']
        for file,key in [('final.pt','final_sha256'),('test.npy','test_sha256'),('selection.json','selection_sha256'),('sampling_audit.json','sampling_audit_sha256')]:
            assert digest(out/file)==done[key]
        frozen[str(out.relative_to(root))]=digest(out/'complete.json')
    assert len(frozen)==36
    with (root/'evaluation_started.json').open('x') as f:json.dump(dict(models=frozen,time=time.time()),f,indent=2)
    baseline=read(root/'baseline_references.json');refs=read(root/'references.json')
    individual=[];summary=[]
    for task in TASKS:
        for beta in BETAS:
            cell=root/'data'/task/f'beta_{beta}';manifest=read(cell/'manifest.json')
            assert digest(cell/'data/test.csv')==manifest['hashes']['data/test.csv']
            test=frame(cell/'data/test.csv')
            np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
            y=test[:,2]
            ordinary=predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean')
            for seed in SEEDS:
                original=baseline[f'{task}/{beta}/{seed}']
                assert digest(Path(original['path'])/'test.npy')==original['test_sha256']
                predictions={'item_mean':ordinary,'LIST_v1':np.load(Path(original['path'])/'test.npy')}
                directories={'LIST_v1':Path(original['path'])}
                for v in archived.VARIANTS:
                    ref=refs[f'{task}/{beta}/{seed}/{v}'];p=Path(ref['path'])
                    assert digest(p/'test.npy')==ref['test_sha256']
                    assert digest(p/'complete.json')==ref['complete_sha256']
                    predictions[v]=np.load(p/'test.npy');directories[v]=p
                for v in VARIANTS:
                    p=output(root,task,v,beta,seed)
                    predictions[v]=np.load(p/'test.npy');directories[v]=p
                metrics={v:metric(pred,y) for v,pred in predictions.items()}
                for v in predictions:
                    choice=dict(epoch=0,lambda_value=0.) if v=='item_mean' else read(directories[v]/'selection.json')
                    row=dict(task=task,beta=beta,seed=seed,variant=v,epoch=choice['epoch'],lambda_value=choice['lambda_value'],**metrics[v])
                    for m in ['mae','rmse']:
                        for ref in ['LIST_v1','raw_point','raw_gaussian','coarse_point','coarse_gaussian']:
                            row[f'delta_{m}_vs_{ref}']=row[m]-metrics[ref][m]
                    individual.append(row)
            for v in [*REFERENCES,*VARIANTS]:
                selected=[x for x in individual if x['task']==task and x['beta']==beta and x['variant']==v]
                row=dict(task=task,beta=beta,variant=v)
                for m in ['mae','rmse']:
                    row[m]=float(np.mean([x[m] for x in selected]))
                    row[m+'_std']=float(np.std([x[m] for x in selected],ddof=1))
                    for ref in ['LIST_v1','raw_point','raw_gaussian','coarse_point','coarse_gaussian']:
                        key=f'delta_{m}_vs_{ref}';row[key]=float(np.mean([x[key] for x in selected]))
                summary.append(row)
    for name,data in [('comparison',summary),('per_seed',individual)]:
        with (root/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save(root/'summary.json',dict(status='PASS',comparison=summary,per_seed=individual))
    save(root/'verification.json',dict(status='PASS',new_models=36,all_final_models_before_test=True,
        all_initial_parameters_match_gaussian=True,unchanged_releases=True,no_hyperparameter_search=True))
    names={'item_mean':'普通均值','LIST_v1':'原版LIST v1','raw_point':'原ID＋成组确定性',
        'raw_gaussian':'原ID＋高斯','coarse_point':'粗码＋成组确定性','coarse_gaussian':'粗码＋高斯',
        'raw_energy':'原ID＋20步郎之万','coarse_energy':'粗码＋20步郎之万'}
    labels={'music_movie_clean':'Music→Movie去重','book_movie':'Book→Movie','book_music':'Book→Music'}
    lines=['# 原ID／粗码条件能量模型：与高斯匹配的三种子多数据集实验','',
        f"Slurm作业{read(root/'protocol.json')['slurm_job']}；36个新模型，固定参数，无调参；高斯、点模型和发布文件直接复用9003。",'',
        '| 任务 | β | 方法 | MAE ± SD | RMSE ± SD | ΔRMSE 对原版LIST |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for x in summary:
        lines.append(f"| {labels[x['task']]} | {x['beta']}% | {names[x['variant']]} | {x['mae']:.6f} ± {x['mae_std']:.6f} | {x['rmse']:.6f} ± {x['rmse_std']:.6f} | {x['delta_rmse_vs_LIST_v1']:+.6f} |")
    lines+=['','共同参数：std初始0.5、下限0.05、上限1、评分噪声方差1；dim32、hidden64、dropout0.1；64完整用户/批、10轮、Adam0.001、weight_decay1e-5；相同排名池化、普通均值、验证epoch/λ规则与从头重训。',
        '能量模型固定：训练/预测各20步MALA，独立步长均0.25，训练32链、预测128链，能量修正上界5。未使用此前单种子寻优结果，也没有调整共享网络宽度。所有初始参数与同输入高斯逐位一致；仅已有能量网络解冻，新增4,225个可训练参数。总参数张量相同，不代表训练容量或计算开销相同。',
        '能量训练objective为近似最大似然梯度代理，不能与高斯NLL数值比较；最终比较MAE/RMSE。接受率只作运行检查，不证明链混合充分。推断只使用源列表条件先验，不使用验证/测试评分构造后验。',
        '粗码发布文件与9003逐字节相同，因此本轮复用其发布层隐私结果；不把能量建模解释为新增隐私保护。历史原版LIST为2048交互批，其余四个9003对照及两个新模型为64用户批。',
        '全部36个新模型和预测冻结后才在本轮读取测试标签；历史测试划分已经用于探索，属于探索性比较。三种子指标取算术平均，不是预测集成。','',
        '[协议](list_coarse_sid_energy_protocol.md) · [完整指标](list_coarse_sid_energy_results.csv) · [逐种子](list_coarse_sid_energy_per_seed_results.csv)']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    for source,target in [('report.md','list_coarse_sid_energy_experiment.md'),('comparison.csv','list_coarse_sid_energy_results.csv'),('per_seed.csv','list_coarse_sid_energy_per_seed_results.csv')]:
        shutil.copyfile(root/source,BASE/'docs'/target)
    print('ENERGY_EXTENSION_EVALUATED',root,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','train','worker','evaluate'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',choices=TASKS);p.add_argument('--beta',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--worker',type=int,default=0)
    a=p.parse_args()
    if a.command=='train':train(a.root,a.task,a.variant,a.beta,a.seed)
    elif a.command=='worker':worker(a.root,a.worker)
    else:globals()[a.command](a.root)


if __name__=='__main__':main()
