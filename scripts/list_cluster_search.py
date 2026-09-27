"""Validation-only search of cluster count for LIST."""
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
VARIANTS=['cooccurrence','size_matched_random']
SEGMENTED=BASE/'runs/list_segmented_8703'

CLUSTER_PARENT=BASE/'runs/list_cluster_8708'
COUNTS=[1,2,3,4,8,16]

class ClusterResidual(ResidualList):
    """Hard item clusters; empty groups and unknown items contribute zero."""
    def __init__(self,lists,vocab,items,base,cfg,groups):
        super().__init__(lists,vocab,items,base,cfg)
        self.cluster_count=cfg['cluster_count']
        if self.cluster_count>1:self.encoder[0]=nn.Linear(self.cluster_count*cfg['dim'],cfg['hidden'])
        self.register_buffer('groups',groups,persistent=False)
    def pool(self,users):
        rows=self.lists[users];groups=self.groups[users];pools=[]
        for j in range(self.cluster_count):
            w=self.rank_weights[None,:]*(groups==j)
            w=w/w.sum(1,keepdim=True).clamp_min(1e-12)
            pools.append(self.source_items(rows,per_sample_weights=w))
        return torch.cat(pools,dim=1)
    def correction(self,u,i):
        z=self.encoder(self.pool(u))*(self.groups[u]>=0).any(1,keepdim=True)
        return (z*self.target_items(i)).sum(-1)


def parameter_hash(model,shared=False):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not shared or not name.startswith('encoder.0.'):
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def cluster_run(root,seed,beta,arm,variant,stage):
    study=root;root=root/variant

    check_code(root);student_guard(root)
    cell,cfg,m=context(root,beta);out=cell/f'seed_{seed}/residual_{arm}'
    assert not (out/f'{stage}_started.json').exists();save(out/f'{stage}_started.json',dict(job=os.environ['SLURM_JOB_ID'],gpu=os.environ['CUDA_VISIBLE_DEVICES']))
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
    tx=None if stage=='select' else torch.tensor(np.load(cell/'data/test_features.npy'),device='cuda',dtype=torch.long)
    selection=read(out/'selection.json') if stage=='refit' else None;init_hashes=[]
    for phase in (['select','refit'] if stage=='both' else [stage]):
        data=frame(cell/('data/inner_train.csv' if phase=='select' else 'data/train.csv'))
        basepath=cell/('inner_baseline' if phase=='select' else 'baseline')
        stats=dict(np.load(basepath/'statistics.npz'))
        base64=predict(stats,np.arange(m['target_items']),'item_mean')
        base=torch.tensor(base64.astype(np.float32),device='cuda');seen=torch.tensor(np.load(basepath/'seen.npy'),device='cuda')
        cluster_dir=study/f'clusters/beta_{beta}/seed_{seed}/{phase}'
        cluster_record=read(cluster_dir/'complete.json')
        assert digest(cluster_dir/'groups.npz')==cluster_record['groups_sha256']
        cluster_data=np.load(cluster_dir/'groups.npz')
        group_rows=torch.full(lists.shape,-1,device='cuda',dtype=torch.int8)
        group_rows[:len(ids)]=torch.tensor(cluster_data[variant][assignment],device='cuda')
        seed_all(seed);model=ClusterResidual(lists,len(vocab),m['target_items'],base,cfg,group_rows).cuda()
        init_hashes.append(parameter_hash(model))
        common_hash=parameter_hash(model,shared=True)
        # Real data dimensions: check every common tensor against the old initialization.
        seed_all(seed);original=ResidualList(lists,len(vocab),m['target_items'],base,cfg).cuda()
        assert common_hash==parameter_hash(original,shared=True)
        parent=read(PARENT/f'beta_{beta}/seed_{seed}/residual_{arm}/complete.json')
        assert parameter_hash(original)==parent['initial_parameters_sha256']
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        parent_trainable=sum(p.numel() for p in original.parameters() if p.requires_grad)
        assert trainable-parent_trainable==(cfg['cluster_count']-1)*cfg['dim']*cfg['hidden']
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
            save(out/'select_complete.json',dict(status='PASS',selection_sha256=digest(out/'selection.json'),
                validation_sha256=digest(out/'validation.npy'),selected_sha256=digest(out/'selected.pt'),
                initial_parameters_sha256=init_hashes[0],shared_initial_parameters_sha256=common_hash,
                trainable_parameters=trainable,parent_trainable_parameters=parent_trainable,
                cluster_select_sha256=cluster_record['groups_sha256'],inner_train_rows=len(data)))
        else:
            assert init_hashes[-1]==read(out/'select_complete.json')['initial_parameters_sha256']
            torch.save(model.state_dict(),out/'final.pt')
            p=base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(model,tx,seen)
            np.save(out/'test.npy',p);save(out/'refit_history.json',history)
            restored=ClusterResidual(lists,len(vocab),m['target_items'],base,cfg,group_rows).cuda()
            restored.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(restored,tx,seen),p)
            save(out/'complete.json',dict(status='PASS',selection_sha256=digest(out/'selection.json'),final_sha256=digest(out/'final.pt'),test_sha256=digest(out/'test.npy'),
                assignment_sha256=digest(out/'assignment.npy'),initial_parameters_sha256=init_hashes[0],
                shared_initial_parameters_sha256=common_hash,trainable_parameters=trainable,parent_trainable_parameters=parent_trainable,
                cluster_refit_sha256=cluster_record['groups_sha256'],cluster_select_sha256=read(study/f'clusters/beta_{beta}/seed_{seed}/select/complete.json')['groups_sha256'],
                full_outer_train_rows=len(data),refit_epochs=epochs,refit_reinitialized=True,restored_predictions=True,test_labels_opened=False))
            del restored
        del model,opt,x,y,data
    print('RESIDUAL_FROZEN',beta,seed,arm,flush=True)


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    cfg=read(CLUSTER_PARENT/'protocol.json')
    assert read(CLUSTER_PARENT/'cluster_verification.json')['status']=='PASS'
    for p,h in cfg['code_hashes'].items():assert digest(Path(p))==h,p
    paths=[Path(__file__),BASE/'scripts/slurm_list_cluster_search.sh']
    cfg=dict(cfg,cluster_counts=COUNTS,cluster_parent=str(CLUSTER_PARENT),
        parent_cluster_protocol_sha256=digest(CLUSTER_PARENT/'protocol.json'),
        search_rule='for each beta choose common C minimizing mean of three seeds best validation RMSE; ties choose smaller C; no test-based choice',
        feature_policy='reuse frozen training-only SVD features from8708 for each seed/beta/phase; change only KMeans cluster count',
        capacity='encoder C*32->64->32; extra parameters versus global pool=(C-1)*2048; C1 retains original first-layer init; C>1 retains original cluster init procedure',
        parameter_change='candidate-specific: (C-1)*2048 additional trainable parameters versus the original global pool',
        frozen_design='validation-only fixed grid of six cluster counts; no learning-rate, SVD-dimension or epoch-budget search',
        final_policy='refit validation-selected C and C1 real lists; selected C also matched-size random real and cooccurrence shuffled controls; C1 random identical hence omitted',
        search_design='fixed grid1,2,3,4,8,16; all candidate validation models and C choices frozen before any new test evaluation',
        cluster_fit='normalized training-only SVD32 features with KMeans at the candidate C',
        pooling='C separate rank-weighted pools concatenated; no extra residual branch',bootstrap_seed=1002,
        code_hashes={**cfg['code_hashes'],**{str(p.resolve()):digest(p) for p in paths}})
    save(root/'protocol.json',cfg)
    for count in COUNTS:
        kroot=root/f'c_{count}';kroot.mkdir();kc=dict(cfg,cluster_count=count);save(kroot/'protocol.json',kc)
        for variant in VARIANTS:
            vr=kroot/variant;vr.mkdir();save(vr/'protocol.json',dict(kc,variant=variant))
            for beta in BETAS:
                cell=vr/f'beta_{beta}';cell.mkdir();old=PARENT/f'beta_{beta}'
                for d in ['data','baseline','inner_baseline']:(cell/d).symlink_to((old/d).resolve(),target_is_directory=True)
                m=read(old/'manifest.json')
                for p,h in m['hashes'].items():assert digest(old/p)==h,p
                save(cell/'manifest.json',m)
                for seed in SEEDS:
                    for arm in ARMS:(cell/f'seed_{seed}/residual_{arm}').mkdir(parents=True)
    print('SEARCH_PREPARED',root,flush=True)


def fit_groups(root,seed,beta):
    from sklearn.cluster import KMeans
    from list_cluster import size_matched_groups
    check_code(root);student_guard(root);cfg=read(root/'protocol.json');count=cfg['cluster_count']
    query=PRIOR/TASK/f'seed_{seed}/queries/DNN/topk_ids.npz';api=np.load(query)
    ids=api['item_ids'];vocab=np.unique(ids);compact=np.searchsorted(vocab,ids)+1
    for phase in ['select','refit']:
        previous=CLUSTER_PARENT/f'clusters/beta_{beta}/seed_{seed}/{phase}'
        record=read(previous/'complete.json');assert digest(previous/'groups.npz')==record['groups_sha256']
        assert digest(query)==record['query_sha256']
        a=np.load(previous/'groups.npz');active=a['active_items'];counts=a['incidence_counts'];latent=a['latent']
        out=root/f'clusters/beta_{beta}/seed_{seed}/{phase}';out.mkdir(parents=True,exist_ok=False)
        if count==3:
            item_cluster=a['item_cluster'];centers=a['centers'];iterations=record['kmeans_iterations']
        else:
            km=KMeans(n_clusters=count,n_init=cfg['kmeans_n_init'],max_iter=cfg['kmeans_max_iter'],random_state=cfg['clustering_seed'])
            raw=km.fit_predict(latent)
            order=sorted(range(count),key=lambda c:(np.log1p(counts[active[raw==c]]).mean(),int(active[raw==c].min())))
            remap=np.empty(count,dtype=np.int8);remap[order]=np.arange(count,dtype=np.int8)
            item_cluster=np.full(len(vocab)+1,-1,dtype=np.int8);item_cluster[active]=remap[raw]
            centers=km.cluster_centers_[order];iterations=int(km.n_iter_)
        groups=item_cluster[compact];random=size_matched_groups(groups,seed)
        for j in range(count):np.testing.assert_array_equal((groups==j).sum(1),(random==j).sum(1))
        np.savez_compressed(out/'groups.npz',item_cluster=item_cluster,cooccurrence=groups,size_matched_random=random,
            incidence_counts=counts,fit_users=a['fit_users'],active_items=active,latent=latent,centers=centers)
        diagnostics={}
        for group in ['inner_train','validation','train','test']:
            users=np.load(PARENT/f'beta_{beta}/data/{group}_users.npy');g=groups[users]
            sizes=np.stack([(g==j).sum(1) for j in range(count)],1)
            diagnostics[group]=dict(unknown_fraction=float((g<0).mean()),mean_group_sizes=sizes.mean(0).tolist(),empty_group_fraction=(sizes==0).mean(0).tolist())
        save(out/'complete.json',dict(status='PASS',cluster_count=count,groups_sha256=digest(out/'groups.npz'),
            parent_groups_sha256=record['groups_sha256'],fit_users_sha256=record['fit_users_sha256'],query_sha256=record['query_sha256'],
            diagnostics=diagnostics,kmeans_iterations=iterations,item_counts=[int((item_cluster==j).sum()) for j in range(count)],
            training_only_features=True,test_labels_opened=False))
    print('CLUSTER_COUNT_READY',count,beta,seed,flush=True)


def selftest(root):
    from list_cluster import ClusterResidual as OldCluster,size_matched_groups
    cfg=read(root/'protocol.json');check_code(root);results=[]
    lists=torch.tensor([list(range(1,101)),list(range(100,0,-1)),[0]*100,list(range(1,101))],device='cuda')
    base=torch.tensor([3.,4.,2.,3.],device='cuda');u=torch.arange(4,device='cuda')
    for count in COUNTS:
        cc=dict(cfg,cluster_count=count);g=np.full((4,100),-1,dtype=np.int8);g[0,:90]=np.arange(90)%count;g[1]=0
        models=[];seed_all(123);original=ResidualList(lists,100,4,base,cc).cuda()
        for groups in [g,size_matched_groups(g,123)]:
            seed_all(123);m=ClusterResidual(lists,100,4,base,cc,torch.tensor(groups,device='cuda')).cuda();m.eval()
            assert parameter_hash(m,shared=True)==parameter_hash(original,shared=True)
            extra=sum(p.numel() for p in m.parameters())-sum(p.numel() for p in original.parameters())
            assert extra==(count-1)*2048
            torch.testing.assert_close(m(u,u),base,rtol=0,atol=0)
            h=m.pool(u).reshape(4,count,32)
            for j in range(count):
                w=m.rank_weights[None,:]*(m.groups==j);w=w/w.sum(1,keepdim=True).clamp_min(1e-12)
                torch.testing.assert_close(h[:,j],(F.embedding(lists,m.source_items.weight)*w[:,:,None]).sum(1))
            models.append(m)
        assert parameter_hash(models[0])==parameter_hash(models[1])
        if count==1:assert parameter_hash(models[0])==parameter_hash(original)
        if count==3:
            seed_all(123);old=OldCluster(lists,100,4,base,cc,models[0].groups).cuda();old.eval()
            assert parameter_hash(old)==parameter_hash(models[0])
            torch.testing.assert_close(old.pool(u),models[0].pool(u),rtol=0,atol=0)
        for m in models:
            (m(u,u)-5).square().mean().backward();assert m.encoder[-1].weight.grad.abs().sum()>0
            with torch.no_grad():m.encoder[-1].weight.normal_(std=.1);m.encoder[-1].bias.fill_(1.)
            torch.testing.assert_close(m(u,u)[2:],base[2:],rtol=0,atol=0)
            m.zero_grad();m(u,u).square().sum().backward()
            for j in range(count):assert m.source_items.weight.grad[lists[0,m.groups[0]==j]].abs().sum()>0
        results.append(dict(cluster_count=count,extra_parameters=extra,status='PASS'))
    save(root/'selftest.json',dict(status='PASS',counts=results,empty_unknown_fallback=True,C1_original_init=True,C3_old_pool_exact=True))
    print('SEARCH_SELFTEST_PASS',flush=True)


def choose(root,beta):
    check_code(root);student_guard(root);rows=[];frozen={}
    for count in COUNTS:
        vals=[]
        for seed in SEEDS:
            out=root/f'c_{count}/cooccurrence/beta_{beta}/seed_{seed}/residual_real'
            c=read(out/'select_complete.json');assert c['status']=='PASS'
            for file,key in [('selection.json','selection_sha256'),('validation.npy','validation_sha256'),('selected.pt','selected_sha256')]:assert digest(out/file)==c[key]
            sel=read(out/'selection.json');vals.append(sel['validation_rmse'])
            frozen[str(out.relative_to(root))]=digest(out/'select_complete.json')
        rows.append(dict(cluster_count=count,mean_validation_rmse=float(np.mean(vals)),std_validation_rmse=float(np.std(vals,ddof=1)),per_seed=dict(zip(map(str,SEEDS),vals))))
    best=min(rows,key=lambda r:(r['mean_validation_rmse'],r['cluster_count']))
    save(root/f'choice_beta_{beta}.json',dict(status='PASS',beta=beta,cluster_count=best['cluster_count'],grid=rows,frozen_validation_models=frozen,test_labels_opened=False))
    print('CHOSEN_BY_VALIDATION',beta,best,flush=True)


def final_plan(root,beta):
    chosen=read(root/f'choice_beta_{beta}.json')['cluster_count']
    runs=[(c,'cooccurrence','real') for c in sorted({1,chosen})]
    runs.append((chosen,'cooccurrence','shuffled'))
    if chosen!=1:runs.append((chosen,'size_matched_random','real'))
    return runs


def finals(root,beta):
    import contextlib
    for count,variant,arm in final_plan(root,beta):
        for seed in SEEDS:
            out=root/f'c_{count}/{variant}/beta_{beta}/seed_{seed}/residual_{arm}'
            stage='refit' if (out/'select_complete.json').exists() else 'both'
            with (out/'final_train.log').open('w') as log,contextlib.redirect_stdout(log):
                cluster_run(root/f'c_{count}',seed,beta,arm,variant,stage)
            print('FINAL_MODEL_FROZEN',beta,count,variant,arm,seed,flush=True)

def evaluate(root):
    check_code(root);cfg=read(root/'protocol.json');assert read(root/'selftest.json')['status']=='PASS'
    assert digest(CLUSTER_PARENT/'protocol.json')==cfg['parent_cluster_protocol_sha256']
    frozen={};audit=[]
    for beta in BETAS:
        cell=PARENT/f'beta_{beta}';val=frame(cell/'data/validation.csv')
        inner_mean=predict(dict(np.load(cell/'inner_baseline/statistics.npz')),val[:,1].astype(int),'item_mean')
        baseline_rmse=float(np.sqrt(np.square(inner_mean-val[:,2]).mean()))
        choice=read(root/f'choice_beta_{beta}.json');means={}
        for count in COUNTS:
            values=[]
            for seed in SEEDS:
                out=root/f'c_{count}/cooccurrence/beta_{beta}/seed_{seed}/residual_real'
                c=read(out/'select_complete.json');rel=str(out.relative_to(root))
                assert digest(out/'select_complete.json')==choice['frozen_validation_models'][rel]
                assert c['trainable_parameters']-c['parent_trainable_parameters']==(count-1)*2048
                history=read(out/'select_history.json');assert [x['epoch'] for x in history]==list(range(1,11))
                expected=(baseline_rmse,0,0.)
                for h in history:
                    for lam in cfg['lambdas']:
                        rmse=h['validation'][str(lam)]
                        if rmse<expected[0]:expected=(rmse,h['epoch'],lam)
                sel=read(out/'selection.json')
                assert sel['epoch']==expected[1] and sel['lambda_value']==expected[2]
                assert abs(sel['validation_rmse']-expected[0])<1e-12
                assert abs(np.sqrt(np.square(np.load(out/'validation.npy')-val[:,2]).mean())-expected[0])<1e-12
                for file,key in [('selection.json','selection_sha256'),('validation.npy','validation_sha256'),('selected.pt','selected_sha256')]:assert digest(out/file)==c[key]
                values.append(expected[0]);audit.append(dict(beta=beta,seed=seed,cluster_count=count,epoch=expected[1],lambda_value=expected[2],validation_rmse=expected[0]))
            means[count]=float(np.mean(values))
        assert choice['cluster_count']==min(COUNTS,key=lambda c:(means[c],c))
        for row in choice['grid']:assert row['mean_validation_rmse']==means[row['cluster_count']]
        for count,variant,arm in final_plan(root,beta):
            for seed in SEEDS:
                out=root/f'c_{count}/{variant}/beta_{beta}/seed_{seed}/residual_{arm}'
                c=read(out/'complete.json');selected=read(out/'select_complete.json');sel=read(out/'selection.json')
                assert c['status']=='PASS' and c['refit_reinitialized'] and c['restored_predictions'] and not c['test_labels_opened']
                assert c['initial_parameters_sha256']==selected['initial_parameters_sha256']
                peer=read(root/f'c_{count}/cooccurrence/beta_{beta}/seed_{seed}/residual_real/select_complete.json')
                assert c['initial_parameters_sha256']==peer['initial_parameters_sha256']
                assert c['trainable_parameters']-c['parent_trainable_parameters']==(count-1)*2048
                assert len(read(out/'refit_history.json'))==sel['epoch']==c['refit_epochs']
                assert c['full_outer_train_rows']==len(frame(cell/'data/train.csv'))
                for file,key in [('selection.json','selection_sha256'),('test.npy','test_sha256'),('final.pt','final_sha256'),('assignment.npy','assignment_sha256')]:assert digest(out/file)==c[key]
                assert digest(out/'assignment.npy')==digest(cell/f'seed_{seed}/residual_{arm}/assignment.npy')
                assert (root/f'choice_beta_{beta}.json').stat().st_mtime_ns<(out/'complete.json').stat().st_mtime_ns
                for phase in ['select','refit']:
                    p=root/f'c_{count}/clusters/beta_{beta}/seed_{seed}/{phase}';a=np.load(p/'groups.npz');record=read(p/'complete.json')
                    assert c[f'cluster_{phase}_sha256']==digest(p/'groups.npz')==record['groups_sha256']
                    previous=CLUSTER_PARENT/f'clusters/beta_{beta}/seed_{seed}/{phase}'
                    old=np.load(previous/'groups.npz');assert record['parent_groups_sha256']==digest(previous/'groups.npz')
                    np.testing.assert_array_equal(a['latent'],old['latent']);np.testing.assert_array_equal(a['fit_users'],old['fit_users'])
                    distances=np.square(a['latent'][:,None,:]-a['centers'][None,:,:]).sum(-1)
                    np.testing.assert_array_equal(distances.argmin(1),a['item_cluster'][a['active_items']])
                    np.testing.assert_array_equal(a['cooccurrence']<0,a['size_matched_random']<0)
                    for j in range(count):np.testing.assert_array_equal((a['cooccurrence']==j).sum(1),(a['size_matched_random']==j).sum(1))
                frozen[str(out.relative_to(root))]=digest(out/'complete.json')
    save(root/'validation_audit.json',dict(status='PASS',candidates=len(audit),rows=audit))
    with (root/'evaluation_started.json').open('x') as f:json.dump(dict(final_models=frozen,choices={str(b):digest(root/f'choice_beta_{b}.json') for b in BETAS}),f,indent=2)
    results={};rows=[];per_seed=[]
    for beta in BETAS:
        cell=PARENT/f'beta_{beta}';manifest=read(cell/'manifest.json')
        assert digest(cell/'data/test.csv')==manifest['hashes']['data/test.csv']
        test=frame(cell/'data/test.csv');y=test[:,2];_,users=np.unique(test[:,0],return_inverse=True)
        np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
        mean=predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean')
        refs={'item_mean':mean[None,:],
            'global_LIST':np.stack([np.load(cell/f'seed_{s}/residual_real/test.npy') for s in SEEDS]),
            'previous_C3':np.stack([np.load(CLUSTER_PARENT/f'cooccurrence/beta_{beta}/seed_{s}/residual_real/test.npy') for s in SEEDS])}
        selected=read(root/f'choice_beta_{beta}.json')['cluster_count']
        new={}
        for count,variant,arm in final_plan(root,beta):
            name=f'C{count}_{variant}_{arm}'
            new[name]=np.stack([np.load(root/f'c_{count}/{variant}/beta_{beta}/seed_{seed}/residual_{arm}/test.npy') for seed in SEEDS])
        controls={'C1':new['C1_cooccurrence_real'],'shuffled':new[f'C{selected}_cooccurrence_shuffled']}
        if selected!=1:controls['size_matched_random']=new[f'C{selected}_size_matched_random_real']
        models={}
        for name,p in {**refs,**new}.items():
            assert p.shape[1]==len(y) and np.isfinite(p).all()
            met=[metrics(row,y,users) for row in p]
            agg={k:dict(mean=float(np.mean([x[k] for x in met])),std=float(np.std([x[k] for x in met],ddof=1)) if len(met)>1 else 0.) for k in ['mae','rmse']}
            comparisons={}
            if name==f'C{selected}_cooccurrence_real':
                comparisons={ref:paired_seed_mean(p,q,y,users,cfg) for ref,q in {**refs,**controls}.items()}
            models[name]=dict(aggregate=agg,comparisons=comparisons,per_seed=met)
            rows.append(dict(beta=beta,method=name,selected_cluster_count=selected,mae=agg['mae']['mean'],mae_std=agg['mae']['std'],rmse=agg['rmse']['mean'],rmse_std=agg['rmse']['std']))
            for i,m in enumerate(met):per_seed.append(dict(beta=beta,method=name,seed=SEEDS[i],mae=m['mae'],rmse=m['rmse']))
        results[str(beta)]=dict(selected_cluster_count=selected,models=models)
    save(root/'summary.json',dict(status='PASS',results=results,protocol=cfg))
    for name,data in [('comparison',rows),('per_seed',per_seed)]:
        with (root/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save(root/'verification.json',dict(status='PASS',validation_candidates=len(audit),final_models=len(frozen),choice_uses_validation_only=True,
        chosen_controls_same_initialization=True,cluster_count_parameter_formula_verified=True,train_only_frozen_SVD_features=True,
        refit_same_initialization=True,all_models_frozen_before_test=True,refit_model_restore_matches=True))
    lines=['# LIST 簇数搜索结果','',
        '固定搜索C=1/2/3/4/8/16；每档β按三个种子的平均验证RMSE选择同一个C；每个种子的epoch与lambda仍由验证集选择。仅选中C、单簇及选中C的对照进入新测试评价；历史测试已使用，结果为探索性。',
        '向量32、隐藏64、学习率0.001、batch2048、最多10轮等保持不变。新增参数=(C−1)×2048；簇数变化伴随容量变化。所有C复用8708对应训练阶段的SVD特征；未知源物品不参与聚合。C1使用原全局模型初始化，但未知ID处理不同。','',
        '| C | β20% 验证RMSE | β80% 验证RMSE | 比全局增加参数 |','| --- | ---: | ---: | ---: |']
    choices={b:read(root/f'choice_beta_{b}.json') for b in BETAS}
    for idx,count in enumerate(COUNTS):lines.append(f"| {count} | {choices[20]['grid'][idx]['mean_validation_rmse']:.6f} | {choices[80]['grid'][idx]['mean_validation_rmse']:.6f} | {(count-1)*2048} |")
    lines+=['',f"验证集选择：β20% C={choices[20]['cluster_count']}；β80% C={choices[80]['cluster_count']}。",'',
        '| β | 方法 | MAE ± 标准差 | RMSE ± 标准差 |','| --- | --- | ---: | ---: |']
    for row in rows:lines.append(f"| {row['beta']}% | {row['method']} | {row['mae']:.6f} ± {row['mae_std']:.6f} | {row['rmse']:.6f} ± {row['rmse_std']:.6f} |")
    lines+=['','## 选中模型相对参照的配对差异','',
        '负值有利于选中模型；95%区间按用户重采样2000次，条件于当前模型和划分，未校正多重比较。','',
        '| β | 参照 | ΔMAE [95%区间] | ΔRMSE [95%区间] |','| --- | --- | --- | --- |']
    for beta,d in results.items():
        selected=d['models'][f"C{d['selected_cluster_count']}_cooccurrence_real"]
        for ref,c in selected['comparisons'].items():
            values=[f"{c['delta_'+k]:+.6f} [{c[k+'_95ci'][0]:+.6f}, {c[k+'_95ci'][1]:+.6f}]" for k in ['mae','rmse']]
            lines.append('| '+beta+'% | '+ref+' | '+' | '.join(values)+' |')
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('CLUSTER_SEARCH_ALL_PASS',json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','selftest','fit_groups','train','choose','finals','evaluate'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--count',type=int,choices=COUNTS)
    p.add_argument('--beta',type=int,choices=BETAS);p.add_argument('--seed',type=int,choices=SEEDS)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='fit_groups':fit_groups(root/f'c_{a.count}',a.seed,a.beta)
    elif a.stage=='train':cluster_run(root/f'c_{a.count}',a.seed,a.beta,'real','cooccurrence','select')
    elif a.stage in ['choose','finals']:globals()[a.stage](root,a.beta)
    else:globals()[a.stage](root)
