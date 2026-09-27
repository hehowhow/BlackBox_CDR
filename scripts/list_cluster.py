"""Training-list co-occurrence clustering with per-user size-matched random controls."""
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

class ClusterResidual(ResidualList):
    """Hard item clusters; empty groups and unknown items contribute zero."""
    def __init__(self,lists,vocab,items,base,cfg,groups):
        super().__init__(lists,vocab,items,base,cfg)
        self.encoder[0]=nn.Linear(3*cfg['dim'],cfg['hidden'])
        self.register_buffer('groups',groups,persistent=False)
    def pool(self,users):
        rows=self.lists[users];groups=self.groups[users];pools=[]
        for j in range(3):
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


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    parent_cfg=read(PARENT/'protocol.json')
    assert read(PARENT/'summary.json')['status']==read(PARENT/'verification.json')['status']=='PASS'
    for p,h in parent_cfg['code_hashes'].items():assert digest(Path(p))==h,p
    paths=[Path(__file__),BASE/'scripts/slurm_list_cluster.sh']
    for p,h in read(SEGMENTED/'protocol.json')['code_hashes'].items():assert digest(Path(p))==h,p
    paths += [BASE/'scripts/verify_list_cluster.py',BASE/'scripts/verify_list_segmented.py']
    import sklearn,scipy
    cfg=dict(parent_cfg,variants=VARIANTS,parent=str(PARENT),parent_protocol_sha256=digest(PARENT/'protocol.json'),
        segmented_parent=str(SEGMENTED),segmented_summary_sha256=digest(SEGMENTED/'summary.json'),
        clustering_seed=20260925,cluster_count=3,svd_dim=32,svd_iterations=7,kmeans_n_init=10,kmeans_max_iter=300,
        cluster_fit='current training overlap users only; item x user rank-weighted incidence, item-row L2 normalization, TruncatedSVD32, latent-row L2 normalization, KMeans3',
        cluster_order='ascending mean log1p(training list incidence), then smallest compact item ID for ties; ordering uses no labels',
        refit_cluster_policy='refit SVD and KMeans using full outer TRAIN lists; independently refit LIST from same initialization for validation-selected epochs',
        unknown_policy='training-unseen source item has group -1 and zero pooling weight; empty groups zero; no known source items gives zero residual',
        pooling='three separately normalized absolute-rank-weighted pools concatenated; MLP96->64->32',
        residual='ordinary item mean plus cluster-list correction; no extra residual branch; frozen item bias; zero last encoder layer',
        fair_control='permute cluster assignments among known positions separately for each user; exact per-user cluster counts and unknown mask preserved; no change to item order or rank weights',
        parameter_change='4096 additional parameters versus global LIST; same parameter count and initialization as 8703 segmented LIST',
        frozen_design='single clustering recipe fixed before new evaluation; no cluster count, feature or learning-rate search; historical test reused',
        bootstrap_seed=1001,library_versions=dict(sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__,torch=torch.__version__),
        code_hashes={**parent_cfg['code_hashes'],**{str(p.resolve()):digest(p) for p in paths}})
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
    print('CLUSTER_PREPARED',root,flush=True)


def size_matched_groups(groups,seed):
    rng=np.random.default_rng(seed+9100);random=groups.copy()
    for row in random:
        valid=np.flatnonzero(row>=0)
        row[valid]=rng.permutation(row[valid])
    return random


def fit_groups(root,seed,beta):
    from scipy import sparse
    from scipy.stats import spearmanr
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    check_code(root);student_guard(root)
    cfg=read(root/'protocol.json');cell=PARENT/f'beta_{beta}'
    query=PRIOR/TASK/f'seed_{seed}/queries/DNN'
    assert digest(query/'topk_ids.npz')==read(query/'complete.json')['item_ids_sha256']
    api=np.load(query/'topk_ids.npz');ids=api['item_ids']
    np.testing.assert_array_equal(api['user_ids'],np.arange(len(ids)))
    vocab=np.unique(ids);compact=np.searchsorted(vocab,ids)+1
    q=(1/np.log2(np.arange(cfg['k'])+2)).astype(np.float32)
    for phase,group in [('select','inner_train'),('refit','train')]:
        out=root/f'clusters/beta_{beta}/seed_{seed}/{phase}';out.mkdir(parents=True,exist_ok=False)
        fit_users=np.load(cell/f'data/{group}_users.npy')
        # Only these columns enter SVD/KMeans. All other lists are transformed later by ID lookup.
        training=compact[fit_users]
        incidence=sparse.csr_matrix((np.tile(q,len(fit_users)),(training.reshape(-1),np.repeat(np.arange(len(fit_users)),cfg['k']))),shape=(len(vocab)+1,len(fit_users)))
        counts=np.bincount(training.reshape(-1),minlength=len(vocab)+1)
        active=np.flatnonzero(counts);x=normalize(incidence[active],norm='l2',axis=1)
        svd=TruncatedSVD(n_components=cfg['svd_dim'],n_iter=cfg['svd_iterations'],random_state=cfg['clustering_seed'])
        latent=normalize(svd.fit_transform(x),norm='l2',axis=1)
        km=KMeans(n_clusters=3,n_init=cfg['kmeans_n_init'],max_iter=cfg['kmeans_max_iter'],random_state=cfg['clustering_seed'])
        raw=km.fit_predict(latent)
        order=sorted(range(3),key=lambda c:(np.log1p(counts[active[raw==c]]).mean(),int(active[raw==c].min())))
        remap=np.empty(3,dtype=np.int8);remap[order]=np.arange(3,dtype=np.int8)
        item_cluster=np.full(len(vocab)+1,-1,dtype=np.int8);item_cluster[active]=remap[raw]
        groups=item_cluster[compact];random=size_matched_groups(groups,seed)
        for j in range(3):np.testing.assert_array_equal((groups==j).sum(1),(random==j).sum(1))
        np.testing.assert_array_equal(groups<0,random<0)
        np.savez_compressed(out/'groups.npz',item_cluster=item_cluster,cooccurrence=groups,size_matched_random=random,
            incidence_counts=counts,fit_users=fit_users,active_items=active,latent=latent,centers=km.cluster_centers_[order],svd_components=svd.components_)
        diagnostics={}
        for name in ['inner_train','validation','train','test']:
            users=np.load(cell/f'data/{name}_users.npy');g=groups[users]
            sizes=np.stack([(g==j).sum(1) for j in range(3)],1)
            diagnostics[name]=dict(users=len(users),unknown_fraction=float((g<0).mean()),all_unknown_users=int((g<0).all(1).sum()),
                mean_group_sizes=sizes.mean(0).tolist(),empty_group_fraction=(sizes==0).mean(0).tolist())
        frequency_summary=[]
        for j in range(3):
            n=counts[item_cluster==j];frequency_summary.append(dict(cluster=j,items=len(n),incidences=int(n.sum()),
                incidence_median=float(np.median(n)),incidence_mean=float(n.mean())))
        # Frequency-only threshold clustering is a diagnostic, never used by the predictor.
        logc=np.log1p(counts[active]);total=float(np.square(logc-logc.mean()).sum())
        within=sum(float(np.square(logc[item_cluster[active]==j]-logc[item_cluster[active]==j].mean()).sum()) for j in range(3))
        correlations=[float(spearmanr(latent[:,j],logc).statistic) for j in range(cfg['svd_dim'])]
        save(out/'complete.json',dict(status='PASS',fit_group=group,fit_users_sha256=digest(cell/f'data/{group}_users.npy'),
            query_sha256=digest(query/'topk_ids.npz'),groups_sha256=digest(out/'groups.npz'),fitted_users=len(fit_users),
            observed_items=len(active),unknown_items=int((item_cluster[1:]<0).sum()),diagnostics=diagnostics,cluster_frequencies=frequency_summary,
            log_incidence_variance_explained_by_clusters=1-within/total,
            max_absolute_latent_log_incidence_spearman=float(np.max(np.abs(correlations))),
            svd_explained_variance=float(svd.explained_variance_ratio_.sum()),kmeans_iterations=int(km.n_iter_),
            test_labels_opened=False,rank_feature='1/log2(rank+1)',matched_random_counts=True))
        print('CLUSTERS_FROZEN',beta,seed,phase,json.dumps(frequency_summary),flush=True)


def selftest(root):
    cfg=read(root/'protocol.json');check_code(root)
    lists=torch.tensor([list(range(1,101)),list(range(100,0,-1)),[0]*100,list(range(1,101))],device='cuda')
    g=np.array([[0]*10+[1]*20+[2]*60+[-1]*10,[1]*100,[-1]*100,[-1]*100],dtype=np.int8)
    baseline=torch.tensor([3.,4.,2.,3.],device='cuda');u=torch.arange(4,device='cuda');models=[]
    seed_all(123);original=ResidualList(lists,100,4,baseline,cfg).cuda()
    for groups in [g,size_matched_groups(g,123)]:
        seed_all(123);m=ClusterResidual(lists,100,4,baseline,cfg,torch.tensor(groups,device='cuda')).cuda();m.eval()
        assert parameter_hash(m,shared=True)==parameter_hash(original,shared=True)
        assert sum(p.numel() for p in m.parameters())-sum(p.numel() for p in original.parameters())==4096
        torch.testing.assert_close(m(u,u),baseline,rtol=0,atol=0)
        pools=m.pool(u).reshape(4,3,cfg['dim']);reconstructed=torch.zeros_like(pools[:,0])
        for j in range(3):
            weights=m.rank_weights[None,:]*(m.groups==j);mass=weights.sum(1,keepdim=True)
            w=weights/mass.clamp_min(1e-12)
            manual=(F.embedding(lists,m.source_items.weight)*w[:,:,None]).sum(1)
            torch.testing.assert_close(pools[:,j],manual)
            reconstructed+=pools[:,j]*mass
        known=m.rank_weights[None,:]*(m.groups>=0)
        torch.testing.assert_close(reconstructed,original.source_items(lists,per_sample_weights=known))
        (m(u,u)-5).square().mean().backward();assert m.encoder[-1].weight.grad.abs().sum()>0
        models.append(m)
    assert parameter_hash(models[0])==parameter_hash(models[1])
    assert not torch.allclose(models[0].pool(u),models[1].pool(u))
    for m in models:
        with torch.no_grad():m.encoder[-1].weight.normal_(std=.1);m.encoder[-1].bias.fill_(1.)
        torch.testing.assert_close(m(u,u)[2:],baseline[2:],rtol=0,atol=0)
        m.zero_grad();m(u,u).square().sum().backward()
        assert torch.isfinite(m.source_items.weight.grad).all()
        for j in range(3):assert m.source_items.weight.grad[lists[0,m.groups[0]==j]].abs().sum()>0
    save(root/'selftest.json',dict(status='PASS',manual_pooling_match=True,empty_groups_zero=True,
        unknown_source_fallback=True,matched_random_counts=True,zero_initial_correction=True,all_groups_receive_gradient=True,
        identical_variant_initialization=True,shared_initialization_matches_parent=True,added_parameters=4096))
    print('CLUSTER_SELFTEST_PASS',flush=True)


def cluster_run(root,seed,beta,arm,variant):
    study=root;root=root/variant

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


def evaluate(root):
    check_code(root);cfg=read(root/'protocol.json')
    assert read(root/'selftest.json')['status']=='PASS'
    assert digest(SEGMENTED/'summary.json')==cfg['segmented_summary_sha256']
    cluster_hashes={}
    for beta in BETAS:
        for seed in SEEDS:
            for phase in ['select','refit']:
                directory=root/f'clusters/beta_{beta}/seed_{seed}/{phase}'
                record=read(directory/'complete.json')
                assert record['status']=='PASS'
                assert digest(directory/'groups.npz')==record['groups_sha256']
                cluster_hashes[str(directory.relative_to(root))]=digest(directory/'complete.json')
    save(root/'cluster_freeze.json',cluster_hashes)
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
              'size_matched_random_real':np.stack([np.load(root/'size_matched_random'/f'beta_{beta}/seed_{s}/residual_real/test.npy') for s in SEEDS]),
              'rank_segments_real':np.stack([np.load(SEGMENTED/'rank_segments'/f'beta_{beta}/seed_{s}/residual_real/test.npy') for s in SEEDS])}
        # Parent rows are carried forward unchanged, with provenance.
        previous=parent_summary['results'][str(beta)]
        b=previous['baseline'];rows.append(dict(beta=beta,method='item_mean',mae=b['mae'],mae_std=0.,rmse=b['rmse'],rmse_std=0.,origin='8650'))
        for name,v in previous['models'].items():
            a=v['aggregate'];rows.append(dict(beta=beta,method=name,mae=a['mae']['mean'],mae_std=a['mae']['std'],rmse=a['rmse']['mean'],rmse_std=a['rmse']['std'],origin='8650'))
        for name,v in read(SEGMENTED/'summary.json')['results'][str(beta)].items():
            a=v['aggregate'];rows.append(dict(beta=beta,method=name,mae=a['mae']['mean'],mae_std=a['mae']['std'],rmse=a['rmse']['mean'],rmse_std=a['rmse']['std'],origin='8703'))
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
                rows.append(dict(beta=beta,method=name,mae=aggregate['mae']['mean'],mae_std=aggregate['mae']['std'],rmse=aggregate['rmse']['mean'],rmse_std=aggregate['rmse']['std'],origin='cluster'))
                for s,v in per_seed.items():
                    sel=read(root/variant/f'beta_{beta}/seed_{s}/residual_{arm}/selection.json')
                    seed_rows.append(dict(beta=beta,variant=variant,arm=arm,seed=s,epoch=sel['epoch'],lambda_value=sel['lambda_value'],mae=v['mae'],rmse=v['rmse']))
        results[str(beta)]=models
    save(root/'summary.json',dict(status='PASS',protocol=cfg,results=results,parent_summary_sha256=digest(PARENT/'summary.json')))
    for name,data in [('comparison',rows),('per_seed',seed_rows)]:
        with (root/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save(root/'verification.json',dict(status='PASS',new_models=24,same_common_initial_parameters_as_rank_control=True,identical_initialization_between_new_variants=True,extra_parameters=4096,same_derangements_as_rank_control=True,all_frozen_before_test=True))
    lines=['# LIST 共现聚类聚合实验','',
        'Music→Movie去重版；β为测试冷用户比例；三种种子；均值±样本标准差，越低越好。',
        '训练用户Top100构建物品×用户的排名加权稀疏矩阵，物品行L2归一化，SVD32降维后再次行归一化，KMeans聚为三类。选择阶段只拟合inner_train，最终重训重新拟合outer_train。未知源ID不参与聚合，空簇为零。',
        'cooccurrence按物品簇分组；size_matched_random在每个用户内打乱已知物品的簇标签，严格保持每用户三组大小及未知位置。两个新模型具有相同初始化、参数量和训练流程；与排名分段同为96→64→32，比全局池化多4096参数。',
        '仅使用源API列表ID及排名，无真实源评分、教师权重或源内容；训练文件访问保护禁止测试标签。24个模型全部冻结后统一评价。未调簇数、降维或学习率；沿用相同验证选epoch和lambda、全量重训、错配用户列表及三个种子。',
        '历史测试已使用，本轮为探索。固定随机对照保留了聚类产生的组大小信息；它检验组内物品归属的作用，不是完全去除聚类信息的对照。','',
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
    (root/'report.md').write_text('\n'.join(lines));print('CLUSTER_ALL_PASS',json.dumps(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','selftest','fit_groups','train','evaluate']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--beta',type=int,choices=BETAS);p.add_argument('--arm',choices=ARMS);p.add_argument('--variant',choices=VARIANTS)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='train':cluster_run(root,a.seed,a.beta,a.arm,a.variant)
    elif a.stage=='fit_groups':fit_groups(root,a.seed,a.beta)
    else:globals()[a.stage](root)
