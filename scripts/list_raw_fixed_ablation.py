"""Fixed paper parameters; raw-ID vs coarse-code Gaussian and MALA ablation."""
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
from book_movie_protocol import BASE,digest,save,predict
from list_source_backbones import read,check_code
from small_task_comparison import frame
from list_ambiguity_story import parameter_hash,metric
from list_sid_story import old_baseline
from list_raw_fixed_ablation_model import matched_pair,shared_hash
import list_coarse_sid_study as archived

G=BASE/'runs/list_coarse_sid_9003';E=BASE/'runs/list_coarse_sid_energy_9011'
TASKS,BETAS,SEEDS=archived.TASKS,archived.BETAS,archived.SEEDS
output=archived.output
FIXED=dict(k=100,dim=32,hidden=64,dropout=.1,epochs=10,lr=.001,weight_decay=1e-5,user_batch_size=64,
    eval_user_batch=128,latent_std=.5,logvar_bounds=[math.log(.05**2),0.],observation_variance=1.,
    lambdas=[0.,.25,.5,.75,1.],train_steps=20,eval_steps=20,train_mala_step=.25,eval_mala_step=.25,
    train_chains=32,eval_chains=128,energy_bound=5.)


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    for parent in [G,E]:
        assert read(parent/'independent_review.json')['status']=='PASS';check_code(parent)
    cfg=read(E/'protocol.json')
    assert all(cfg[k]==v for k,v in FIXED.items())
    cfg.update(study='Fixed paper hyperparameters; remove coarse publication only; downstream initialization aligned',
        variants=['raw_energy'],parent=str(E),gaussian_parent=str(G),policy='fixed_ablation',
        initialization='All non-source energy tensors initialized identically to paper coarse model; raw Gaussian active tensors already match',
        unchanged='Exact archived9003training loop and9011energy class; 64full users/batch; originalV1rank pool,MLP,item scorer,means and validation rule',
        input='Same DNN Top100 and rank positions; published raw IDs instead of many-to-one coarse codes; original raw lookup std.1',
        references='18rawGaussians and18coarseGaussians from9003;18coarseMALA and18legacyrawMALA from9011; originalV1',
        tuning='None; all hyperparameters copied from paper coarse models, including initialstd.5 for every cell',
        selection='Same epoch0..10 andlambda grid; final chosen epoch/lambda may differ by validation; no forced equality of chosen values',
        interpretation='Raw source table has more rows; input detail and parameter sharing differ; no equal-capacity claim',
        slurm_job=os.environ.get('SLURM_JOB_ID'))
    additions=['list_raw_fixed_ablation.py','list_raw_fixed_ablation_model.py','verify_list_raw_fixed_ablation.py','slurm_list_raw_fixed_ablation.sh']
    for name in additions:cfg['code_hashes'][str(BASE/'scripts'/name)]=digest(BASE/'scripts'/name)
    save(root/'protocol.json',cfg)
    (root/'source_snapshot').mkdir()
    for p in cfg['code_hashes']:shutil.copy2(p,root/'source_snapshot'/Path(p).name)
    for name in ['data','releases']:(root/name).symlink_to((G/name).resolve(),target_is_directory=True)
    refs={};init_records={}
    for task in TASKS:
        manifest=read(root/'data'/task/'beta_20/manifest.json')
        for seed in SEEDS:
            meta=read(archived.release_dir(root,task,20,seed,'select')/'release.json')
            for family,parent in [('gaussian',G),('energy',E)]:
                dummy=torch.zeros((1,100),dtype=torch.long);base=torch.zeros(manifest['target_items'])
                model,rec=matched_pair(dummy,meta['raw_vocab_size'],meta['coarse_vocab_size'],manifest['target_items'],base,cfg,seed,family)
                for beta in BETAS:
                    for kind,key in [('raw','legacy_raw_initial_sha256'),('coarse','coarse_initial_sha256')]:
                        done=read(output(parent,task,kind+'_'+family,beta,seed)/'complete.json')
                        assert all(done['phases'][phase]['initial_sha256']==rec[key] for phase in ['select','refit'])
                assert rec['trainable_shared_match']
                if family=='energy':assert set(rec['original_differing_trainable_parameters'])=={'energy_net.0.weight','energy_net.0.bias'}
                else:assert not rec['original_differing_trainable_parameters']
                init_records[f'{task}/{seed}/{family}']=rec;del model
            for beta in BETAS:
                output(root,task,'raw_energy',beta,seed).mkdir(parents=True)
                paths={'raw_gaussian':output(G,task,'raw_gaussian',beta,seed),'coarse_gaussian':output(G,task,'coarse_gaussian',beta,seed),
                    'coarse_energy':output(E,task,'coarse_energy',beta,seed),'raw_energy_legacy':output(E,task,'raw_energy',beta,seed),
                    'LIST_v1':old_baseline(task,beta,seed),'raw_point64':output(G,task,'raw_point',beta,seed)}
                refs[f'{task}/{beta}/{seed}']={}
                for name,path in paths.items():
                    done=read(path/'complete.json');assert done['status']=='PASS'
                    for filename,key in [('final.pt','final_sha256'),('test.npy','test_sha256'),('selection.json','selection_sha256')]:assert digest(path/filename)==done[key]
                    refs[f'{task}/{beta}/{seed}'][name]=dict(path=str(path),complete_sha256=digest(path/'complete.json'),test_sha256=done['test_sha256'],selection_sha256=done['selection_sha256'])
    save(root/'references.json',refs);save(root/'initialization_audit.json',dict(status='PASS',records=init_records))
    save(root/'tasks.json',[dict(task=t,beta=b,seed=s,variant='raw_energy') for s in SEEDS for t in TASKS for b in BETAS])
    save(root/'plan_frozen.json',dict(time=time.time(),new_energy_models=18,reused_raw_gaussian_models=18,
        references_sha256=digest(root/'references.json'),tasks_sha256=digest(root/'tasks.json'),initialization_sha256=digest(root/'initialization_audit.json')))
    print('FIXED_ABLATION_PREPARED',root,flush=True)


def train(root,task,variant,beta,seed):
    import list_coarse_sid_model as factory
    check_code(root);assert variant=='raw_energy' and os.environ.get('SLURM_JOB_ID')
    rec=read(root/'initialization_audit.json')['records'][f'{task}/{seed}/energy'];records=[];acceptance=[]
    original=(factory.make_model,archived.parameter_hash,archived.target_guard)
    def build(lists,vocab,items,base,cfg,family):
        assert family=='energy';phase=['select','refit'][len(records)]
        meta=read(archived.release_dir(root,task,beta,seed,phase)/'release.json')
        model,actual=matched_pair(lists,vocab,meta['coarse_vocab_size'],items,base,cfg,seed,'energy')
        assert actual==rec
        records.append(dict(phase=phase,**actual));native=model.objective
        def objective(*args,**kwargs):
            loss,extra=native(*args,**kwargs);a,b=float(extra['prior_accept']),float(extra['posterior_accept'])
            assert 0<=a<=1 and 0<=b<=1;acceptance.append((phase,a,b));return loss,extra
        model.objective=objective
        return model
    def checked_hash(model,common=False):
        assert not common
        assert shared_hash(model)==rec['coarse_shared_sha256']
        assert parameter_hash(model)==rec['raw_initial_sha256']
        return parameter_hash(model)
    def guard(study,arm,out):
        def extra(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0])).resolve()
            if {'trusted','id_maps','tiger_sid_content'}.intersection(p.parts) or p.name=='sid.npz':raise RuntimeError(f'Raw ablation forbidden read: {p}')
        sys.addaudithook(extra);return original[2](study,arm,out)
    factory.make_model=build;archived.parameter_hash=checked_hash;archived.target_guard=guard
    try:archived.train(root,task,variant,beta,seed)
    finally:factory.make_model,archived.parameter_hash,archived.target_guard=original
    out=output(root,task,variant,beta,seed);done=read(out/'complete.json');coarse=read(output(E,task,'coarse_energy',beta,seed)/'complete.json')
    raw_old=read(output(E,task,'raw_energy',beta,seed)/'complete.json')
    for phase in ['select','refit']:
        p=done['phases'][phase];c=coarse['phases'][phase]
        for k in ['common_initial_sha256','overlap_train_rows','grouped_users','full_target_train_rows','fit_users_sha256']:assert p[k]==c[k]
        assert p['trainable_parameters']-c['trainable_parameters']==(p['vocab_size']-c['vocab_size'])*32
        assert p['release_sha256']==raw_old['phases'][phase]['release_sha256']
    save(out/'sampling_audit.json',dict(status='PASS',initialization_checks=records,
        controls={k:read(root/'protocol.json')[k] for k in ['train_steps','eval_steps','train_mala_step','eval_mala_step','train_chains','eval_chains','energy_bound']},
        acceptance={p:dict(batches=sum(ph==p for ph,a,b in acceptance),
            prior=float(np.mean([a for ph,a,b in acceptance if ph==p])) if any(ph==p for ph,a,b in acceptance) else None,
            posterior=float(np.mean([b for ph,a,b in acceptance if ph==p])) if any(ph==p for ph,a,b in acceptance) else None) for p in ['select','refit']}))
    done.update(non_source_initialization_matches_coarse=True,sampling_audit_sha256=digest(out/'sampling_audit.json'))
    save(out/'complete.json',done)


def worker(root,index):
    tasks=read(root/'tasks.json');announced=False
    while True:
        todo=[t for t in tasks if not (output(root,**t)/'claim.json').exists()]
        if not todo:return
        if (root/'worker_failure.json').exists():raise RuntimeError('Another worker failed')
        free,usage=archived.gpu_available()
        if not free:
            if not announced:print('WAIT_ALLOCATED_GPU',usage,flush=True);announced=True
            time.sleep(10);continue
        for t in todo:
            out=output(root,**t)
            try:
                with (out/'claim.json').open('x') as f:json.dump(dict(worker=index,gpu=os.environ['CUDA_VISIBLE_DEVICES']),f)
            except FileExistsError:continue
            command=[sys.executable,'-u',str(Path(__file__)),'train','--root',str(root)]
            for k,v in t.items():command+=['--'+k,str(v)]
            print('FIXED_ABLATION_START',t,flush=True)
            try:
                with (out/'train.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
            except Exception as error:save(root/'worker_failure.json',dict(task=t,error=str(error)));raise
            break


def csv_write(path,rows):
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def evaluate(root):
    check_code(root);plan=read(root/'plan_frozen.json')
    for n,k in [('references.json','references_sha256'),('tasks.json','tasks_sha256'),('initialization_audit.json','initialization_sha256')]:assert digest(root/n)==plan[k]
    frozen={}
    for task in read(root/'tasks.json'):
        out=output(root,**task);done=read(out/'complete.json')
        assert done['non_source_initialization_matches_coarse'] and not done['test_labels_opened']
        for f,k in [('final.pt','final_sha256'),('test.npy','test_sha256'),('selection.json','selection_sha256'),('sampling_audit.json','sampling_audit_sha256')]:assert digest(out/f)==done[k]
        frozen[str(out.relative_to(root))]=digest(out/'complete.json')
    assert len(frozen)==18
    save(root/'evaluation_started.json',dict(time=time.time(),models=frozen))
    refs=read(root/'references.json');individual=[];summary=[];paired=[]
    for task in TASKS:
        for beta in BETAS:
            cell=root/'data'/task/f'beta_{beta}';manifest=read(cell/'manifest.json')
            assert digest(cell/'data/test.csv')==manifest['hashes']['data/test.csv']
            test=frame(cell/'data/test.csv');np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
            mean=predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean')
            for seed in SEEDS:
                paths={'raw_energy':output(root,task,'raw_energy',beta,seed)}
                for name,ref in refs[f'{task}/{beta}/{seed}'].items():
                    path=Path(ref['path']);assert digest(path/'complete.json')==ref['complete_sha256'];assert digest(path/'test.npy')==ref['test_sha256'];paths[name]=path
                for name in [*paths,'item_mean']:
                    p=mean if name=='item_mean' else np.load(paths[name]/'test.npy')
                    assert np.isfinite(p).all() and p.shape==(len(test),)
                    sel=dict(epoch=0,lambda_value=0.) if name=='item_mean' else read(paths[name]/'selection.json')
                    individual.append(dict(task=task,beta=beta,seed=seed,method=name,**metric(p,test[:,2]),epoch=sel['epoch'],lambda_value=sel['lambda_value'],source_path=str(paths[name]) if name!='item_mean' else str(cell/'baseline/statistics.npz')))
            for name in [*paths,'item_mean']:
                rows=[r for r in individual if r['task']==task and r['beta']==beta and r['method']==name]
                summary.append(dict(task=task,beta=beta,method=name,**{k:float(np.mean([r[k] for r in rows])) for k in ['mae','rmse']},**{k+'_std':float(np.std([r[k] for r in rows],ddof=1)) for k in ['mae','rmse']}))
            for family in ['gaussian','energy']:
                for seed in SEEDS:
                    vals={r['method']:r for r in individual if r['task']==task and r['beta']==beta and r['seed']==seed}
                    paired.append(dict(task=task,beta=beta,seed=seed,family=family,**{'delta_'+k:vals['raw_'+family][k]-vals['coarse_'+family][k] for k in ['mae','rmse']}))
    for name,rows in [('per_seed',individual),('results',summary),('paired',paired)]:
        csv_write(root/f'{name}.csv',rows);shutil.copy2(root/f'{name}.csv',BASE/'docs'/f'list_raw_fixed_ablation_{name}.csv')
    names={'music_movie_clean':'Music→Movie去重','book_movie':'Book→Movie','book_music':'Book→Music'}
    def lookup(t,b,m):return next(r for r in summary if r['task']==t and r['beta']==b and r['method']==m)
    def pair(r):return f'{r["mae"]:.6f} / {r["rmse"]:.6f}'
    lines=['# 固定论文参数下原ID与粗码的消融对照','',
        '按用户明确选择，本轮所有高斯和郎之万均沿用论文粗码版固定超参数，只把发布的粗码换回原物品ID。三数据集、β20/80、三个种子。高斯18个原ID结果复用已核验存档；郎之万18个原ID模型重新训练，以同时对齐能量隐藏层的初始权重。',
        '', '全部设置：DNN Top100、32维、隐藏宽度64、dropout0.1、64完整用户一批、Adam学习率0.001、weight decay1e-5、10轮预算。初始std统一0.5、范围0.05–1、评分噪声方差1。郎之万训练／预测20步，步长0.25／0.25，链数32／128，能量上界5。',
        '', '沿用相同验证集和epoch／λ选择规则，随后从头refit；规则相同不要求最终选中的epoch和λ相同。源物品嵌入表因原ID／粗码词表不同而改变大小与参数共享，不宣称参数总量完全相等。旧原ID郎之万与粗码版的能量隐藏层初始随机值不同；本轮将所有非源嵌入参数初始化逐位对齐，并保留旧结果供追溯。',
        '', '## 原ID与粗码的直接对照','', '数值为三种子平均MAE / RMSE，越低越好。β是冷测试用户占重叠用户的比例。','',
        '| 数据集 | β | 原ID高斯 | 粗码高斯 | 原ID郎之万 | 粗码郎之万 |','|---|---:|---|---|---|---|']
    for t in TASKS:
        for b in BETAS:lines.append('| '+' | '.join([names[t],str(b)+'%',*[pair(lookup(t,b,m)) for m in ['raw_gaussian','coarse_gaussian','raw_energy','coarse_energy']]])+' |')
    lines+=['','## 删除粗码后的指标变化','', 'Δ=原ID−粗码，负数表示原ID误差更低。','', '| 数据集 | β | 高斯ΔMAE | 高斯ΔRMSE | 郎之万ΔMAE | 郎之万ΔRMSE |','|---|---:|---:|---:|---:|---:|']
    for t in TASKS:
        for b in BETAS:
            delta=[f'{lookup(t,b,"raw_"+f)[k]-lookup(t,b,"coarse_"+f)[k]:+.6f}' for f in ['gaussian','energy'] for k in ['mae','rmse']]
            lines.append('| '+' | '.join([names[t],str(b)+'%',*delta])+' |')
    lines+=['','## 原版与历史结果参照','', '| 数据集 | β | 原LIST V1 | 原ID高斯 | 原ID郎之万本轮 | 原ID郎之万旧版 | 普通均值 |','|---|---:|---|---|---|---|---|']
    for t in TASKS:
        for b in BETAS:lines.append('| '+' | '.join([names[t],str(b)+'%',*[pair(lookup(t,b,m)) for m in ['LIST_v1','raw_gaussian','raw_energy','raw_energy_legacy','item_mean']]])+' |')
    lines+=['','原LIST V1采用2048交互批，仅作总体效果参照。本轮原ID与粗码两侧均采用64完整用户批。上一轮按历史验证分数选择初始std的高斯实验单独保留于list_v1_raw_final_experiment.md，不混入本次固定参数消融。','',
        '## 均值与标准差','', '| 数据集 | β | 方法 | MAE | RMSE |','|---|---:|---|---:|---:|']
    for r in summary:lines.append(f'| {names[r["task"]]} | {r["beta"]}% | {r["method"]} | {r["mae"]:.6f} ± {r["mae_std"]:.6f} | {r["rmse"]:.6f} ± {r["rmse_std"]:.6f} |')
    (root/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(root/'report.md',BASE/'docs/list_raw_fixed_ablation_experiment.md')
    save(root/'verification.json',dict(status='PASS',new_raw_energy_models=18,reused_raw_gaussian_models=18,per_seed_rows=len(individual),aggregate_rows=len(summary),fixed_paper_hyperparameters=True,downstream_initialization_matches_coarse=True,all_new_models_frozen_before_test=True))
    print('FIXED_ABLATION_EVALUATED',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','worker','evaluate']);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--task');p.add_argument('--variant',default='raw_energy');p.add_argument('--beta',type=int);p.add_argument('--seed',type=int);p.add_argument('--index',type=int,default=0)
    a=p.parse_args()
    if a.action=='train':train(a.root,a.task,a.variant,a.beta,a.seed)
    elif a.action=='worker':worker(a.root,a.index)
    else:globals()[a.action](a.root)
