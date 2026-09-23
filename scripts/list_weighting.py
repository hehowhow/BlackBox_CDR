"""Two preregistered LIST aggregation variants on top of ordinary-mean residuals."""
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
VARIANTS=['inverse_degree','target_attention']

class WeightedResidual(ResidualList):
    def __init__(self,lists,vocab,items,base,cfg,counts,variant):
        super().__init__(lists,vocab,items,base,cfg)
        self.variant=variant
        self.register_buffer('counts',counts,persistent=False)
    def pool(self,users,items):
        rows=self.lists[users]
        target=self.target_items(items)
        if self.variant=='inverse_degree':
            weights=self.rank_weights[None,:]*torch.rsqrt(1+self.counts[rows])
            weights=weights/weights.sum(1,keepdim=True)
            pooled=self.source_items(rows,per_sample_weights=weights)
        else:
            # No extra parameters: target-aware cosine modulates the fixed rank prior.
            # Skip only the expensive embedding lookup for no-list rows; encoder and
            # dropout keep the full batch shape, matching the control's RNG usage.
            active=rows[:,0]!=0
            pooled=target.new_zeros((len(rows),target.shape[1]))
            weights=self.rank_weights[None,:].expand(len(rows),-1).clone()
            if active.any():
                e=F.embedding(rows[active],self.source_items.weight,padding_idx=0)
                cosine=(F.normalize(e,dim=-1,eps=1e-8)*F.normalize(target[active],dim=-1,eps=1e-8)[:,None,:]).sum(-1)
                w=torch.softmax(self.rank_weights.log()[None,:]+cosine,dim=1)
                pooled[active]=(w[:,:,None]*e).sum(1)
                weights[active]=w
        return pooled,target,weights
    def correction(self,u,i):
        h,v,_=self.pool(u,i)
        z=self.encoder(h)*(self.lists[u,:1]!=0)
        return (z*v).sum(-1)


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    parent_cfg=read(PARENT/'protocol.json')
    assert read(PARENT/'summary.json')['status']==read(PARENT/'verification.json')['status']=='PASS'
    for p,h in parent_cfg['code_hashes'].items():assert digest(Path(p))==h,p
    paths=[Path(__file__),BASE/'scripts/slurm_list_weighting.sh']
    cfg=dict(parent_cfg,variants=VARIANTS,parent=str(PARENT),parent_protocol_sha256=digest(PARENT/'protocol.json'),
        weighting_scope='change pooling only; fixed DNN Top100, ordinary mean residual, dimensions, init, optimizer, lambda and epoch selection unchanged',
        inverse_degree='normalized rank_weight/sqrt(1+c_i); c_i from current TRAIN overlap users API lists only; unseen returned item c_i=0; no raw source popularity',
        target_attention='softmax(log(rank_weight)+cosine(source_item_embedding,target_item_embedding)); temperature1; no added parameters; jointly trained embeddings',
        literature={'LightGCN_SIGIR2020':'https://hexiangnan.github.io/papers/sigir20-LightGCN.pdf','DIN_KDD2018':'https://arxiv.org/pdf/1706.06978'},
        attribution='inspired adaptations, not LightGCN/DIN reproductions; returned-list incidence is not exposure propensity or true source interaction popularity',
        frozen_design='both formulas fixed before reading 8650 test results; no weighting hyperparameter search',
        bootstrap_seed=999,code_hashes={**parent_cfg['code_hashes'],**{str(p.resolve()):digest(p) for p in paths}})
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
    save(root/'data_verification.json',dict(status='PASS',same_parent_splits=True,parent=8650,variants_fixed_without_parent_test_metrics=True))
    print('WEIGHTING_PREPARED',root,flush=True)


def selftest(root):
    cfg=read(root/'protocol.json');seed_all(123)
    lists=torch.tensor([[1,2,3],[3,2,1],[0,0,0]],device='cuda')
    baseline=torch.tensor([3.,4.,2.],device='cuda');counts=torch.tensor([0.,100.,4.,1.],device='cuda')
    u=torch.arange(3,device='cuda');i=u.clone();checks=[]
    for variant in VARIANTS:
        m=WeightedResidual(lists,3,3,baseline,cfg,counts,variant).cuda();m.eval()
        torch.testing.assert_close(m(u,i),baseline,rtol=0,atol=0)
        h,v,w=m.pool(u,i)
        torch.testing.assert_close(w.sum(1),torch.ones(3,device='cuda'))
        (m(u,i)-5).square().mean().backward()
        assert m.encoder[-1].weight.grad.abs().sum()>0
        with torch.no_grad():m.encoder[-1].bias.fill_(1.)
        assert m(u,i)[-1]==baseline[-1]
        if variant=='inverse_degree':
            assert w[0,0]/w[0,2]<m.rank_weights[0]/m.rank_weights[2]
            # Equal frequencies reduce to original pooling, including gradients.
            m.counts.fill_(3.)
            with torch.no_grad():m.encoder[-1].weight.normal_(std=.1)
            original=ResidualList(lists,3,3,baseline,cfg).cuda();original.load_state_dict(m.state_dict());original.eval()
            torch.testing.assert_close(m(u,i),original(u,i),rtol=2e-5,atol=2e-5)
            pooled,_,_=m.pool(u,i)
            reference=original.source_items(lists,per_sample_weights=original.rank_weights.expand_as(lists))
            torch.testing.assert_close(pooled,reference,rtol=2e-5,atol=2e-5)
            m.zero_grad();original.zero_grad()
            m(u,i).square().sum().backward();original(u,i).square().sum().backward()
            for (an,ap),(bn,bp) in zip(m.named_parameters(),original.named_parameters()):
                assert an==bn
                if ap.grad is None:assert bp.grad is None
                else:torch.testing.assert_close(ap.grad,bp.grad,rtol=1e-4,atol=2e-5)
        else:
            with torch.no_grad():
                m.source_items.weight.zero_();m.target_items.weight.zero_()
                m.source_items.weight[1,0]=1.;m.source_items.weight[2,1]=1.
                m.target_items.weight[0,0]=1.;m.target_items.weight[1,1]=1.
            _,_,w2=m.pool(torch.tensor([0,0],device='cuda'),torch.tensor([0,1],device='cuda'))
            assert w2[0,0]>w2[1,0] and w2[0,1]<w2[1,1]
        assert set(dict(m.named_parameters()))==set(dict(ResidualList(lists,3,3,baseline,cfg).named_parameters()))
        checks.append(dict(variant=variant,zero_correction=True,can_learn=True,normalized_weights=True,no_list_fallback=True,no_added_parameters=True))
    save(root/'selftest.json',dict(status='PASS',checks=checks));print('WEIGHTING_SELFTEST_PASS',flush=True)


def weighted_run(root,seed,beta,arm,variant):
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
        training_ids=np.load(cell/('data/inner_train_users.npy' if phase=='select' else 'data/train_users.npy'))
        counts=torch.bincount(lists[torch.tensor(training_ids,device='cuda')].reshape(-1),minlength=len(vocab)+1).float()
        counts[0]=0
        independent=np.bincount(compact[assignment[training_ids]].reshape(-1),minlength=len(vocab)+1)
        np.testing.assert_array_equal(counts.cpu().numpy(),independent)
        np.save(out/f'{phase}_list_frequency.npy',counts.cpu().numpy())
        seed_all(seed);model=WeightedResidual(lists,len(vocab),m['target_items'],base,cfg,counts,variant).cuda()
        h=hashlib.sha256()
        for p in model.parameters():h.update(p.detach().cpu().numpy().tobytes())
        init_hashes.append(h.hexdigest())
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
            restored=WeightedResidual(lists,len(vocab),m['target_items'],base,cfg,counts,variant).cuda()
            restored.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(restored,tx,seen),p)
            save(out/'complete.json',dict(status='PASS',selection_sha256=digest(out/'selection.json'),final_sha256=digest(out/'final.pt'),test_sha256=digest(out/'test.npy'),
                assignment_sha256=digest(out/'assignment.npy'),initial_parameters_sha256=init_hashes[0],
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
                    # Same initialization as the first-stage rank-only residual control.
                    parent=read(PARENT/f'beta_{beta}/seed_{seed}/residual_{arm}/complete.json')
                    assert c['initial_parameters_sha256']==parent['initial_parameters_sha256']
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
              'rank_residual':np.stack([np.load(cell/f'seed_{s}/residual_real/test.npy') for s in SEEDS])}
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
                for v in comparisons.values():v['interpretation']='negative favors weighted model; mean of three individual model metrics; user-cluster CI conditional on fitted models and split; no multiplicity correction'
                name=variant+'_'+arm;models[name]=dict(aggregate=aggregate,per_seed=per_seed,comparisons=comparisons)
                rows.append(dict(beta=beta,method=name,mae=aggregate['mae']['mean'],mae_std=aggregate['mae']['std'],rmse=aggregate['rmse']['mean'],rmse_std=aggregate['rmse']['std'],origin='weighting'))
                for s,v in per_seed.items():
                    sel=read(root/variant/f'beta_{beta}/seed_{s}/residual_{arm}/selection.json')
                    seed_rows.append(dict(beta=beta,variant=variant,arm=arm,seed=s,epoch=sel['epoch'],lambda_value=sel['lambda_value'],mae=v['mae'],rmse=v['rmse']))
        results[str(beta)]=models
    save(root/'summary.json',dict(status='PASS',protocol=cfg,results=results,parent_summary_sha256=digest(PARENT/'summary.json')))
    for name,data in [('comparison',rows),('per_seed',seed_rows)]:
        with (root/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save(root/'verification.json',dict(status='PASS',new_models=24,same_initial_parameters_as_rank_control=True,same_derangements_as_rank_control=True,all_frozen_before_test=True))
    lines=['# LIST列表加权：两种方案与完整跨域对照','',
        'Music→Movie去重版；β=20%/80%；三个种子。以下数值为各种子指标的均值±样本标准差。普通均值不平滑。',
        '原有均值、EMCDR/PFCDR、原LIST、rank残差结果来自8650，未经修改。两种新方案只改变列表池化，保持固定均值修正、参数数量、初始化、列表、划分、优化器、选轮数与lambda规则不变。','',
        '| β | 方法 | MAE ± 标准差 | RMSE ± 标准差 |','| --- | --- | ---: | ---: |']
    for r in rows:lines.append(f"| {r['beta']}% | {r['method']} | {r['mae']:.6f} ± {r['mae_std']:.6f} | {r['rmse']:.6f} ± {r['rmse_std']:.6f} |")
    lines+=['','## 方案与文献来源','',
        'A：频次降权。设排名权重q_r=1/log2(r+1)，c_j为当前训练重叠用户的Top100中物品j出现次数，则w_ur∝q_r/sqrt(1+c_j)，每个用户内归一化。内层选模仅用inner_train列表统计；最终重训重新用outer_train列表统计。验证与测试用户不参与频次估计。',
        '该设计借鉴[LightGCN，SIGIR2020](https://hexiangnan.github.io/papers/sigir20-LightGCN.pdf)的度归一化原则。这里没有图传播，统计的是API返回频次，既不是真实源评分流行度，也不是曝光倾向概率；不声称去除了因果偏差。c=0使用加1平滑以保证有限权重，可能给训练未出现的返回ID较大权重，需要由验证结果检验。',
        'B：目标相关加权。w_uir=softmax(log(q_r)+cos(e_source_item,v_target_item))，温度固定为1。不同目标物品获得不同的源列表表示。源/目标向量均是学生自己的参数，未读取教师权重；没有增加可训练参数。',
        '该设计借鉴[DIN，KDD2018](https://arxiv.org/pdf/1706.06978)按目标聚合兴趣的思想，但使用归一化余弦权重，而不是DIN的MLP局部激活和非归一化权重，属于自设计简化方案，不是DIN复现。',
        '两个公式在查看8650测试结果之前固定；不搜索加权超参数、K或学习率。两版均运行真实与打乱列表，对照使用与rank残差相同的初始参数及错配列表。',
        '各模型仍先在相同内层验证集选择epoch0..10及lambda=0/0.25/0.5/0.75/1，再以相同初始化重训全外层训练数据。全部新模型冻结后统一评价。历史测试已使用，本轮为探索；差异区间条件于当前模型及划分，未校正多重比较。','']
    (root/'report.md').write_text('\n'.join(lines));print('WEIGHTING_ALL_PASS',json.dumps(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','selftest','train','evaluate']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--beta',type=int,choices=BETAS);p.add_argument('--arm',choices=ARMS);p.add_argument('--variant',choices=VARIANTS)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='train':weighted_run(root,a.seed,a.beta,a.arm,a.variant)
    else:globals()[a.stage](root)
