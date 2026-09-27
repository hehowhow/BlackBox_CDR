"""Fixed rank segments versus equal-capacity random rank groups for residual LIST."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from small_task_comparison import (PRIOR,TASK,SEEDS,BETAS,ARMS,ResidualList,context,frame,corrections)
from book_movie_protocol import BASE,digest,save,predict
from residual_experiment import seed_all
from list_source_backbones import student_guard,check_code,read
from residual_summary import metrics
from list_target_heads import paired_seed_mean
PARENT=BASE/'runs/small_task_comparison_8650'
VARIANTS=['rank_segments','random_groups']

class SegmentedResidual(ResidualList):
    """Three rank-weighted pools; groups are fixed, not learned or searched."""
    def __init__(self,lists,vocab,items,base,cfg,variant):
        super().__init__(lists,vocab,items,base,cfg)
        # Construct the original model first to preserve all shared initial tensors.
        self.encoder[0]=nn.Linear(3*cfg['dim'],cfg['hidden'])
        self.variant=variant
        groups=cfg['rank_groups'][variant]
        assert sorted(sum(groups,[]))==list(range(cfg['k']))
        for j,positions in enumerate(groups):
            ix=torch.tensor(positions,device=lists.device,dtype=torch.long)
            self.register_buffer(f'group_{j}',ix,persistent=False)
            w=self.rank_weights[ix]
            self.register_buffer(f'weights_{j}',w/w.sum(),persistent=False)
    def pool(self,users):
        rows=self.lists[users]
        return torch.cat([self.source_items(rows[:,getattr(self,f'group_{j}')],
            per_sample_weights=getattr(self,f'weights_{j}').expand(len(rows),-1)) for j in range(3)],dim=1)
    def correction(self,u,i):
        z=self.encoder(self.pool(u))*(self.lists[u,:1]!=0)
        return (z*self.target_items(i)).sum(-1)


def parameter_hash(model,shared=False):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not shared or not name.startswith('encoder.0.'):
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    parent_cfg=read(PARENT/'protocol.json')
    assert read(PARENT/'summary.json')['status']==read(PARENT/'verification.json')['status']=='PASS'
    for p,h in parent_cfg['code_hashes'].items():assert digest(Path(p))==h,p
    paths=[Path(__file__),BASE/'scripts/slurm_list_segmented.sh']
    permutation=np.random.default_rng(20260924).permutation(100).tolist()
    groups={'rank_segments':[list(range(0,10)),list(range(10,30)),list(range(30,100))],
            'random_groups':[permutation[:10],permutation[10:30],permutation[30:]]}
    cfg=dict(parent_cfg,variants=VARIANTS,parent=str(PARENT),parent_protocol_sha256=digest(PARENT/'protocol.json'),
        rank_groups=groups,partition_seed=20260924,
        pooling='three separately normalized rank-weighted pools concatenated; MLP96->64->32',
        residual='fixed ordinary item mean + three-pool source-list interaction; zero final encoder layer; frozen item bias',
        fair_control='random fixed groups of 10/20/70 rank positions; same parameters, initialization, data, batches, optimizer and selection; no grouping search',
        parameter_change='4096 extra first-layer weights versus original global pool; all other initial tensors unchanged',
        weighting_scope='only pooling and encoder input dimension change; no target-conditioned per-item interaction',
        frozen_design='boundaries10/30/100 and one random partition fixed before new test evaluation; historical test already explored',
        bootstrap_seed=1000,code_hashes={**parent_cfg['code_hashes'],**{str(p.resolve()):digest(p) for p in paths}})
    save(root/'protocol.json',cfg)
    for variant in VARIANTS:
        vroot=root/variant;vroot.mkdir();save(vroot/'protocol.json',dict(cfg,variant=variant))
        for beta in BETAS:
            cell=vroot/f'beta_{beta}';cell.mkdir();old=PARENT/f'beta_{beta}'
            for d in ['data','baseline','inner_baseline']:(cell/d).symlink_to((old/d).resolve(),target_is_directory=True)
            m=read(old/'manifest.json')
            for p,h in m['hashes'].items():assert digest(old/p)==h
            save(cell/'manifest.json',m)
            for seed in SEEDS:
                for arm in ARMS:(cell/f'seed_{seed}/residual_{arm}').mkdir(parents=True)
    save(root/'data_verification.json',dict(status='PASS',same_parent_splits=True,parent=8650,design_fixed_before_new_test_evaluation=True))
    print('SEGMENTED_PREPARED',root,flush=True)


def selftest(root):
    cfg=read(root/'protocol.json');check_code(root)
    lists=torch.tensor([list(range(1,101)),list(range(100,0,-1)),[0]*100],device='cuda')
    baseline=torch.tensor([3.,4.,2.],device='cuda')
    u=torch.arange(3,device='cuda');i=u.clone();models=[]
    seed_all(123);original=ResidualList(lists,100,3,baseline,cfg).cuda()
    for variant in VARIANTS:
        seed_all(123);m=SegmentedResidual(lists,100,3,baseline,cfg,variant).cuda();m.eval()
        assert parameter_hash(m,shared=True)==parameter_hash(original,shared=True)
        assert sum(p.numel() for p in m.parameters())-sum(p.numel() for p in original.parameters())==4096
        torch.testing.assert_close(m(u,i),baseline,rtol=0,atol=0)
        pools=m.pool(u).reshape(3,3,cfg['dim'])
        for j in range(3):
            ix=getattr(m,f'group_{j}');w=getattr(m,f'weights_{j}')
            torch.testing.assert_close(w.sum(),torch.ones((),device='cuda'))
            manual=(F.embedding(lists[:,ix],m.source_items.weight)*w[None,:,None]).sum(1)
            torch.testing.assert_close(pools[:,j],manual)
        masses=torch.stack([m.rank_weights[getattr(m,f'group_{j}')].sum() for j in range(3)])
        global_pool=original.source_items(lists,per_sample_weights=original.rank_weights.expand_as(lists))
        torch.testing.assert_close((pools*masses[None,:,None]).sum(1),global_pool)
        (m(u,i)-5).square().mean().backward()
        assert m.encoder[-1].weight.grad.abs().sum()>0
        models.append(m)
    assert parameter_hash(models[0])==parameter_hash(models[1])
    assert not torch.allclose(models[0].pool(u),models[1].pool(u))
    for m in models:
        with torch.no_grad():m.encoder[-1].weight.normal_(std=.1);m.encoder[-1].bias.fill_(1.)
        m.zero_grad();m(u,i).square().sum().backward()
        for j in range(3):
            assert m.source_items.weight.grad[(getattr(m,f'group_{j}')+1)].abs().sum()>0
        assert m(u,i)[-1]==baseline[-1]
    save(root/'selftest.json',dict(status='PASS',normalized_groups=True,manual_pooling_match=True,
        global_pool_reconstruction=True,zero_initial_correction=True,all_groups_receive_gradient=True,
        no_list_fallback=True,identical_variant_initialization=True,shared_initialization_matches_parent=True,
        added_parameters=4096))
    print('SEGMENTED_SELFTEST_PASS',flush=True)


def segmented_run(root,seed,beta,arm,variant):
    root=root/variant

    check_code(root);student_guard(root)
    cell,cfg,m=context(root,beta);out=cell/f'seed_{seed}/residual_{arm}'
    assert not (out/'started.json').exists();save(out/'started.json',dict(job=os.environ['SLURM_JOB_ID'],gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    query=PRIOR/TASK/f'seed_{seed}/queries/DNN';record=read(query/'complete.json')
    assert digest(query/'topk_ids.npz')==record['item_ids_sha256']
    api=np.load(query/'topk_ids.npz');np.testing.assert_array_equal(api['user_ids'],np.arange(m['overlap_users']))
    ids=api['item_ids'];vocab=np.unique(ids);compact=np.searchsorted(vocab,ids)+1
    assignment=np.arange(len(ids));rng=np.random.default_rng(seed+7000)
    if arm=='shuffled':
        for group in ['inner_train','validation','test']:
            users=np.load(cell/f'data/{group}_users.npy');order=rng.permutation(users)
            assignment[order]=np.roll(order,1)
        assert np.all(assignment!=np.arange(len(ids)))
    np.save(out/'assignment.npy',assignment)
    lists=torch.zeros((m['target_users'],cfg['k']),device='cuda',dtype=torch.long)
    lists[:len(ids)]=torch.tensor(compact[assignment],device='cuda')
    val=frame(cell/'data/validation.csv');vx=torch.tensor(val[:,:2],device='cuda',dtype=torch.long)
    tx=torch.tensor(np.load(cell/'data/test_features.npy'),device='cuda',dtype=torch.long)
    selection=None;init_hashes=[]
    for phase in ['select','refit']:
        data=frame(cell/('data/inner_train.csv' if phase=='select' else 'data/train.csv'))
        basepath=cell/('inner_baseline' if phase=='select' else 'baseline')
        stats=dict(np.load(basepath/'statistics.npz'))
        base64=predict(stats,np.arange(m['target_items']),'item_mean')
        base=torch.tensor(base64.astype(np.float32),device='cuda');seen=torch.tensor(np.load(basepath/'seen.npy'),device='cuda')
        seed_all(seed);model=SegmentedResidual(lists,len(vocab),m['target_items'],base,cfg,variant).cuda()
        init_hashes.append(parameter_hash(model))
        common_hash=parameter_hash(model,shared=True)
        # Real data dimensions: check every common tensor against the old initialization.
        seed_all(seed);original=ResidualList(lists,len(vocab),m['target_items'],base,cfg).cuda()
        assert common_hash==parameter_hash(original,shared=True)
        parent=read(PARENT/f'beta_{beta}/seed_{seed}/residual_{arm}/complete.json')
        assert parameter_hash(original)==parent['initial_parameters_sha256']
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        parent_trainable=sum(p.numel() for p in original.parameters() if p.requires_grad)
        assert trainable-parent_trainable==4096
        del original
        seed_all(seed+100)
        x=torch.tensor(data[:,:2],device='cuda',dtype=torch.long);y=torch.tensor(data[:,2],device='cuda',dtype=torch.float32)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        gen=torch.Generator(device='cuda').manual_seed(seed+200)
        if phase=='select':
            np.testing.assert_array_equal(corrections(model,vx,seen),np.zeros(len(vx),dtype=np.float32))
            baseval=base64[val[:,1].astype(int)]
            selection=dict(epoch=0,lambda_value=0.,validation_rmse=float(np.sqrt(np.square(baseval-val[:,2]).mean())))
            torch.save(model.state_dict(),out/'selected.pt');np.save(out/'validation.npy',baseval)
        epochs=cfg['epochs'] if phase=='select' else selection['epoch'];history=[]
        for epoch in range(1,epochs+1):
            tick=time.monotonic();model.train();total=torch.zeros((),device='cuda')
            for ix in torch.randperm(len(x),device='cuda',generator=gen).split(cfg['batch_size']):
                opt.zero_grad(set_to_none=True);loss=(model(x[ix,0],x[ix,1])-y[ix]).square().mean();loss.backward();opt.step();total+=loss.detach()*len(ix)
            row=dict(epoch=epoch,train_mse=total.item()/len(x),seconds=time.monotonic()-tick)
            assert np.isfinite(row['train_mse'])
            if phase=='select':
                correction=corrections(model,vx,seen);row['validation']={}
                for lam in cfg['lambdas']:
                    pred=baseval+lam*correction
                    rmse=float(np.sqrt(np.square(pred-val[:,2]).mean()));row['validation'][str(lam)]=rmse
                    if rmse<selection['validation_rmse']:
                        selection=dict(epoch=epoch,lambda_value=lam,validation_rmse=rmse)
                        torch.save(model.state_dict(),out/'selected.pt');np.save(out/'validation.npy',pred)
            history.append(row);save(out/f'{phase}_history.json',history)
            print('RESIDUAL',beta,seed,arm,phase,json.dumps(row),flush=True)
        if phase=='select':
            model.load_state_dict(torch.load(out/'selected.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(baseval+selection['lambda_value']*corrections(model,vx,seen),np.load(out/'validation.npy'))
            save(out/'selection.json',selection)
        else:
            assert init_hashes[0]==init_hashes[1]
            torch.save(model.state_dict(),out/'final.pt')
            p=base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(model,tx,seen)
            np.save(out/'test.npy',p);save(out/'refit_history.json',history)
            restored=SegmentedResidual(lists,len(vocab),m['target_items'],base,cfg,variant).cuda()
            restored.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(restored,tx,seen),p)
            save(out/'complete.json',dict(status='PASS',selection_sha256=digest(out/'selection.json'),final_sha256=digest(out/'final.pt'),test_sha256=digest(out/'test.npy'),
                assignment_sha256=digest(out/'assignment.npy'),initial_parameters_sha256=init_hashes[0],
                shared_initial_parameters_sha256=common_hash,trainable_parameters=trainable,parent_trainable_parameters=parent_trainable,
                full_outer_train_rows=len(data),refit_epochs=epochs,refit_reinitialized=True,restored_predictions=True,test_labels_opened=False))
            del restored
        del model,opt,x,y,data
    print('RESIDUAL_FROZEN',beta,seed,arm,flush=True)


def evaluate(root):
    check_code(root);cfg=read(root/'protocol.json')
    assert read(root/'selftest.json')['status']=='PASS'
    frozen={}
    for variant in VARIANTS:
        for beta in BETAS:
            for seed in SEEDS:
                for arm in ARMS:
                    p=root/variant/f'beta_{beta}/seed_{seed}/residual_{arm}';c=read(p/'complete.json')
                    assert c['status']=='PASS'
                    for f,k in [('final.pt','final_sha256'),('selection.json','selection_sha256'),('test.npy','test_sha256')]:assert digest(p/f)==c[k]
                    # Same common tensors as parent; all tensors identical across new variants.
                    parent=read(PARENT/f'beta_{beta}/seed_{seed}/residual_{arm}/complete.json')
                    peer=read(root/VARIANTS[0]/f'beta_{beta}/seed_{seed}/residual_real/complete.json')
                    assert c['initial_parameters_sha256']==peer['initial_parameters_sha256']
                    assert c['shared_initial_parameters_sha256']==peer['shared_initial_parameters_sha256']
                    assert c['trainable_parameters']==peer['trainable_parameters']
                    assert c['trainable_parameters']-c['parent_trainable_parameters']==4096
                    assert c['assignment_sha256']==parent['assignment_sha256']
                    frozen[str(p.relative_to(root))]=digest(p/'complete.json')
    with (root/'evaluation_started.json').open('x') as f:json.dump(frozen,f,indent=2)
    rows=[];seed_rows=[];results={}
    parent_summary=read(PARENT/'summary.json')
    for beta in BETAS:
        cell=PARENT/f'beta_{beta}';m=read(cell/'manifest.json')
        assert digest(cell/'data/test.csv')==m['hashes']['data/test.csv']
        test=frame(cell/'data/test.csv');y=test[:,2];_,users=np.unique(test[:,0],return_inverse=True)
        np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
        mean=predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean')
        old=read(cell/'old_list.json')
        refs={'item_mean':mean[None,:],
              'LIST_original':np.stack([np.load(Path(old[str(s)]['path'])/'test.npy') for s in SEEDS]),
              'rank_residual':np.stack([np.load(cell/f'seed_{s}/residual_real/test.npy') for s in SEEDS]),
              'random_groups_real':np.stack([np.load(root/'random_groups'/f'beta_{beta}/seed_{s}/residual_real/test.npy') for s in SEEDS])}
        # Parent rows are carried forward unchanged, with provenance.
        previous=parent_summary['results'][str(beta)]
        b=previous['baseline'];rows.append(dict(beta=beta,method='item_mean',mae=b['mae'],mae_std=0.,rmse=b['rmse'],rmse_std=0.,origin='8650'))
        for name,v in previous['models'].items():
            a=v['aggregate'];rows.append(dict(beta=beta,method=name,mae=a['mae']['mean'],mae_std=a['mae']['std'],rmse=a['rmse']['mean'],rmse_std=a['rmse']['std'],origin='8650'))
        models={}
        for variant in VARIANTS:
            predictions={arm:np.stack([np.load(root/variant/f'beta_{beta}/seed_{s}/residual_{arm}/test.npy') for s in SEEDS]) for arm in ARMS}
            for arm,pred in predictions.items():
                per_seed={str(s):metrics(pred[i],y,users) for i,s in enumerate(SEEDS)}
                aggregate={key:dict(mean=float(np.mean([v[key] for v in per_seed.values()])),std=float(np.std([v[key] for v in per_seed.values()],ddof=1))) for key in ['mae','rmse']}
                comparisons={n:paired_seed_mean(pred,p,y,users,cfg) for n,p in refs.items()}
                if arm=='real':comparisons['shuffled']=paired_seed_mean(pred,predictions['shuffled'],y,users,cfg)
                for v in comparisons.values():v['interpretation']='negative favors this new model; mean of three individual model metrics; user-cluster CI conditional on fitted models and split; no multiplicity correction'
                name=variant+'_'+arm;models[name]=dict(aggregate=aggregate,per_seed=per_seed,comparisons=comparisons)
                rows.append(dict(beta=beta,method=name,mae=aggregate['mae']['mean'],mae_std=aggregate['mae']['std'],rmse=aggregate['rmse']['mean'],rmse_std=aggregate['rmse']['std'],origin='segmented'))
                for s,v in per_seed.items():
                    sel=read(root/variant/f'beta_{beta}/seed_{s}/residual_{arm}/selection.json')
                    seed_rows.append(dict(beta=beta,variant=variant,arm=arm,seed=s,epoch=sel['epoch'],lambda_value=sel['lambda_value'],mae=v['mae'],rmse=v['rmse']))
        results[str(beta)]=models
    save(root/'summary.json',dict(status='PASS',protocol=cfg,results=results,parent_summary_sha256=digest(PARENT/'summary.json')))
    for name,data in [('comparison',rows),('per_seed',seed_rows)]:
        with (root/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save(root/'verification.json',dict(status='PASS',new_models=24,same_common_initial_parameters_as_rank_control=True,identical_initialization_between_new_variants=True,extra_parameters=4096,same_derangements_as_rank_control=True,all_frozen_before_test=True))
    lines=['# LIST分段聚合实验','',
        'Music→Movie去重版；β是测试冷用户占重叠用户的比例；三种模型种子。数值为各模型指标的均值±样本标准差，越低越好。',
        '原有均值、EMCDR/PFCDR、原LIST和rank残差引用8650的冻结结果。rank_segments按1–10/11–30/31–100名分组；random_groups使用预先固定的随机排名分组，组大小同为10/20/70。两版各增加4096个参数，全部初始参数完全相同；与旧模型共享的初始参数保持不变。',
        '各组沿用绝对排名权重1/log2(rank+1)，在组内归一化；三个32维向量拼接后经96→64→32编码，与目标物品向量点积作为普通物品均值的修正。此实验不是逐条目与目标物品交互。',
        '固定DNN Top100、数据划分、学习率0.001、weight decay1e-5、dropout0.1、batch2048、最多10轮；相同验证集选择epoch和lambda后全量重训。未搜索分段边界。真实与错配列表均完整重训，24个新模型全部冻结后才打开测试标签。',
        '随机分组控制参数量和多组结构，但仅测试一个固定随机划分；不能单独排除组内归一化等改变带来的作用。历史测试已多次使用，本轮是探索性结果。','',
        '| β | 方法 | MAE ± 标准差 | RMSE ± 标准差 |','| --- | --- | ---: | ---: |']
    for r in rows:lines.append(f"| {r['beta']}% | {r['method']} | {r['mae']:.6f} ± {r['mae_std']:.6f} | {r['rmse']:.6f} ± {r['rmse_std']:.6f} |")
    lines+=['','## 配对差异（新方法减参照）','',
        '95%区间按测试用户成组重采样2000次，先计算各种子指标再求均值；条件于已训练模型与当前划分，未做多重比较校正。负值代表新方法更好。','',
        '| β | 新方法 | 参照 | ΔRMSE | 95%区间 |','| --- | --- | --- | ---: | --- |']
    for beta,models in results.items():
        for name,result in models.items():
            if not name.endswith('_real'):continue
            for ref,c in result['comparisons'].items():
                lo,hi=c['rmse_95ci'];lines.append(f"| {beta}% | {name} | {ref} | {c['delta_rmse']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    (root/'report.md').write_text('\n'.join(lines));print('SEGMENTED_ALL_PASS',json.dumps(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','selftest','train','evaluate']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--beta',type=int,choices=BETAS);p.add_argument('--arm',choices=ARMS);p.add_argument('--variant',choices=VARIANTS)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='train':segmented_run(root,a.seed,a.beta,a.arm,a.variant)
    else:globals()[a.stage](root)
