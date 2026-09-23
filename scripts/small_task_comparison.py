"""Clean Music->Movie: ordinary mean, old LIST, residual LIST, EMCDR/PFCDR.
Frozen inner selection + full outer refit; no test-label reads during training.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import csv
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import softmax
from book_movie_protocol import BASE, digest, save, predict
from residual_experiment import seed_all, guard_test_labels
from list_source_backbones import ListStudent, student_guard, check_code, read
from list_target_heads import paired_seed_mean
from residual_summary import metrics
import pfcdr_backbone_study as pf

PRIOR=BASE/'runs/paper_list_dnn_8638'
TASK='music_movie_clean'
SEEDS=[2020,2021,2022]
BETAS=[20,80]
BACKBONES=['MF','DNN']
ARMS=['real','shuffled']


def frame(p): return pd.read_csv(p,header=None).to_numpy()

def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    prior=read(PRIOR/'protocol.json')
    for p,h in prior['code_hashes'].items(): assert digest(Path(p))==h,p
    files=['small_task_comparison.py','slurm_small_task_comparison.sh','pfcdr_backbone_study.py',
           'list_source_backbones.py','list_target_heads.py','book_movie_protocol.py',
           'residual_experiment.py','residual_summary.py','run_list_encoder.py']
    paths=[BASE/'scripts'/p for p in files]+[BASE/'external/PFCDR'/p for p in ['models.py','float_embedding.py','inversion.py','inversion_config.json']]
    cfg=dict(task=TASK,seeds=SEEDS,betas=BETAS,backbones=BACKBONES,arms=ARMS,seed=2020,
        dim=32,hidden=64,dropout=.1,k=100,epochs=10,lr=.001,weight_decay=1e-5,batch_size=2048,
        source_dim=10,pretrain_epochs=10,pretrain_lr=.01,pretrain_weight_decay=0.,source_batch=512,target_batch=256,
        source_policy='DNN source checkpoint and ID lists reused exactly from 8638; fresh matched MF; source batch512 kept for DNN comparability (author task3 uses256)',
        inversion_iterations=300,inversion_batch=200,inversion_lr=1e-4,inversion_weight_decay=1e-4,inversion_target=5,inversion_warmup=10,
        bridge_epochs=10,bridge_lr=.005,bridge_batch=1024,bridge_weight_decay=0.,primary_k=70000,
        emcdr_epochs=10,emcdr_lr=.01,emcdr_batch=128,emcdr_weight_decay=0.,
        residual='fixed ordinary item mean + source-list interaction only; no learned item bias; zero final encoder layer; same original LIST encoder/dimensions',
        lambdas=[0.,.25,.5,.75,1.],residual_training='fit unscaled correction at lambda1; validation chooses epoch0..10 and lambda; zero ties keep epoch0',
        selection='inner cold-user validation RMSE; all methods freshly refit full outer train for selected epochs; lambda frozen before refit',
        shuffle='fixed user derangement separately within original inner_train,validation,test groups; unchanged during full refit',
        baselines='ordinary item mean only; all models fallback to ordinary mean on unseen target items',
        pf_protocol='author MF and DNN forward, inversion300 and K70000; all target user rows including cold untrained rows vote; no LR/K search',
        evaluation='every new final model frozen before test labels; historical test reused; comparisons exploratory; no ten-run paper reproduction',
        bootstrap_repeats=2000,bootstrap_seed=998,
        code_hashes={str(p.resolve()):digest(p) for p in paths})
    save(root/'protocol.json',cfg)
    manifest=read(PRIOR/TASK/'manifest.json')
    assert manifest['overlap_users']==17904 and manifest['cross_domain_raw_item_overlap']==0
    save(root/'manifest.json',manifest)
    (root/'source_data').symlink_to(PRIOR/TASK/'source_data',target_is_directory=True)
    for seed in SEEDS:
        for name in BACKBONES: (root/f'sources/seed_{seed}/{name}').mkdir(parents=True)
    for beta in BETAS:
        old=PRIOR/TASK/f'beta_{beta}'; cell=root/f'beta_{beta}';cell.mkdir()
        m=read(old/'manifest.json')
        for p,h in m['hashes'].items(): assert digest(old/p)==h,p
        for d in ['data','baseline','inner_baseline']:(cell/d).symlink_to(old/d,target_is_directory=True)
        save(cell/'manifest.json',m)
        refs={}
        for seed in SEEDS:
            old_model=old/f'seed_{seed}'; record=read(old_model/'complete.json')
            assert digest(old_model/'test.npy')==record['test_sha256']
            refs[str(seed)]=dict(path=str(old_model),complete_sha256=digest(old_model/'complete.json'),test_sha256=record['test_sha256'])
            for name in BACKBONES:(cell/f'seed_{seed}/{name}').mkdir(parents=True)
            for arm in ARMS:(cell/f'seed_{seed}/residual_{arm}').mkdir(parents=True)
        save(cell/'old_list.json',refs)
    save(root/'data_verification.json',dict(status='PASS',prior_job=8638,overlap_users=17904,clean_data_unchanged=True,
             identical_outer_and_inner_splits=True,unchanged_old_list_predictions=True))
    print('PREPARED',root,flush=True)


def context(root,beta):
    cell=root/f'beta_{beta}'
    cfg=read(root/'protocol.json');m=read(cell/'manifest.json')
    for p,h in m['hashes'].items():
        if p!='data/test.csv':assert digest(cell/p)==h,p
    return cell,cfg,m


def inv_domain(out,model,name,domain,ids,cfg,seed):
    for p in model.parameters():p.requires_grad_(False)
    model.eval(); rng=np.random.RandomState(seed+5000+(domain=='tgt'))
    v=pf.table(model,name,domain).iid_embedding.weight.detach()
    all_proto=[]; history=[]
    for start in range(0,len(ids),cfg['inversion_batch']):
        tick=time.monotonic(); u=torch.tensor(ids[start:start+cfg['inversion_batch']],device='cuda',dtype=torch.long)
        with torch.no_grad(): coeff=pf.effective_users(model,name,domain,u)@v.T
        init=softmax(rng.randn(len(u),len(v)),axis=1).astype(np.float32)
        w=torch.tensor(init,device='cuda',requires_grad=True)
        opt=torch.optim.Adam([w],lr=cfg['inversion_lr'],weight_decay=cfg['inversion_weight_decay'])
        best_loss=float('inf');best=None
        for it in range(cfg['inversion_iterations']):
            lr=cfg['inversion_lr']*(it+1)/cfg['inversion_warmup'] if it<cfg['inversion_warmup'] else .5*(1+np.cos(np.pi*(it-cfg['inversion_warmup'])/(cfg['inversion_iterations']-cfg['inversion_warmup'])))*cfg['inversion_lr']
            opt.param_groups[0]['lr']=lr
            loss=((w*coeff).sum(1)-cfg['inversion_target']).square().mean();value=loss.item()
            assert np.isfinite(value)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if value<best_loss or it==1:best=w.detach().clone();best_loss=value
        with torch.no_grad():proto=best@v
        assert torch.isfinite(proto).all()
        all_proto.append(proto.cpu().numpy())
        history.append(dict(start=start,users=len(u),selected_pre_step_mse=best_loss,seconds=time.monotonic()-tick))
        save(out/f'inversion_{domain}_history.json',history)
        print('INVERSION',name,domain,start+len(u),'/',len(ids),flush=True)
        del w,opt,best,coeff
    np.savez(out/f'prototypes_{domain}.npz',user_ids=ids,prototypes=np.concatenate(all_proto))


def train_domain(model,name,domain,data,batch,cfg,seed,out):
    for p in model.parameters():p.requires_grad_(False)
    module=getattr(model,domain+'_model')
    for p in module.parameters():p.requires_grad_(True)
    x=torch.tensor(data[:,:2],device='cuda',dtype=torch.long);y=torch.tensor(data[:,2],device='cuda',dtype=torch.float32)
    opt=torch.optim.Adam(module.parameters(),lr=cfg['pretrain_lr'],weight_decay=0.)
    generator=torch.Generator(device='cuda').manual_seed(seed+4000+(domain=='tgt'))
    history=[]
    for epoch in range(1,cfg['pretrain_epochs']+1):
        tick=time.monotonic();model.train();total=torch.zeros((),device='cuda')
        for idx in torch.randperm(len(x),device='cuda',generator=generator).split(batch):
            opt.zero_grad(set_to_none=True);loss=(model(x[idx],'train_'+domain)-y[idx]).square().mean()
            loss.backward();opt.step();total+=loss.detach()*len(idx)
        row=dict(epoch=epoch,train_mse=total.item()/len(x),seconds=time.monotonic()-tick);history.append(row)
        assert np.isfinite(row['train_mse']);save(out/f'pretrain_{domain}_history.json',history)
        print('PRETRAIN',name,domain,json.dumps(row),flush=True)


def source(root,seed,name):
    check_code(root);guard_test_labels(root)
    cfg=read(root/'protocol.json');m=read(root/'manifest.json');out=root/f'sources/seed_{seed}/{name}'
    assert not (out/'started.json').exists()
    save(out/'started.json',dict(job=os.environ['SLURM_JOB_ID'],gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    seed_all(seed);model=pf.make_model(name,m,10);pf.initialize(model,name,seed)
    if name=='DNN':
        old=PRIOR/TASK/f'seed_{seed}/queries/DNN'
        record=read(old/'complete.json');assert digest(old/'source.pt')==record['source_sha256']
        model.src_model.load_state_dict(torch.load(old/'source.pt',map_location='cuda',weights_only=True))
        save(out/'source_origin.json',dict(reused=str(old/'source.pt'),sha256=record['source_sha256']))
    else:
        assert digest(root/'source_data/source.csv')==m['hashes']['data/source.csv']
        train_domain(model,name,'src',frame(root/'source_data/source.csv'),cfg['source_batch'],cfg,seed,out)
    torch.save(model.src_model.state_dict(),out/'source.pt')
    inv_domain(out,model,name,'src',np.arange(m['overlap_users']),cfg,seed)
    save(out/'complete.json',dict(status='PASS',source_sha256=digest(out/'source.pt'),prototypes_sha256=digest(out/'prototypes_src.npz')))


@torch.no_grad()
def em_predict(model,x,seen,base):
    model.eval()
    raw=torch.cat([model(b,'test_map').reshape(-1) for b in x.split(4096)])
    return torch.where(seen[x[:,1]],raw,base[x[:,1]]).cpu().numpy()


def baseline_run(root,seed,beta,name):
    check_code(root);guard_test_labels(root)
    cell,cfg,m=context(root,beta);out=cell/f'seed_{seed}/{name}'
    assert not (out/'started.json').exists();save(out/'started.json',dict(job=os.environ['SLURM_JOB_ID'],gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    src=root/f'sources/seed_{seed}/{name}';sr=read(src/'complete.json')
    assert digest(src/'source.pt')==sr['source_sha256'] and digest(src/'prototypes_src.npz')==sr['prototypes_sha256']
    spdata=np.load(src/'prototypes_src.npz');np.testing.assert_array_equal(spdata['user_ids'],np.arange(m['overlap_users']))
    sp=torch.tensor(spdata['prototypes'],device='cuda')
    selections={}
    for phase in ['select','refit']:
        d=out/phase;d.mkdir()
        seed_all(seed);model=pf.make_model(name,m,10);pf.initialize(model,name,seed)
        model.src_model.load_state_dict(torch.load(src/'source.pt',map_location='cuda',weights_only=True))
        data=frame(cell/('data/inner_train.csv' if phase=='select' else 'data/train.csv'))
        tids=np.load(cell/('data/inner_train_users.npy' if phase=='select' else 'data/train_users.npy'))
        assert not np.intersect1d(np.unique(data[:,0]),np.load(cell/'data/test_users.npy')).size
        if phase=='select':assert not np.intersect1d(tids,np.load(cell/'data/validation_users.npy')).size
        train_domain(model,name,'tgt',data,cfg['target_batch'],cfg,seed,d)
        torch.save(model.tgt_model.state_dict(),d/'target.pt')
        for p in model.parameters():p.requires_grad_(False)
        basepath=cell/('inner_baseline' if phase=='select' else 'baseline')
        base=torch.tensor(np.load(basepath/'values.npy'),device='cuda');seen=torch.tensor(np.load(basepath/'seen.npy'),device='cuda')
        if phase=='select':
            val=frame(cell/'data/validation.csv');vx=torch.tensor(val[:,:2],device='cuda',dtype=torch.long)
            eval_ids=np.load(cell/'data/validation_users.npy')
        else:
            vx=torch.tensor(np.load(cell/'data/test_features.npy'),device='cuda',dtype=torch.long)
            eval_ids=np.load(cell/'data/test_users.npy')
        tid=torch.tensor(tids,device='cuda',dtype=torch.long)
        # Official EMCDR maps effective DNN user vectors (raw vectors for MF).
        su=pf.effective_users(model,name,'src',tid).detach();tu=pf.effective_users(model,name,'tgt',tid).detach()
        native_s,native_t=model(tid[:8],'train_map')
        torch.testing.assert_close(native_s,model.mapping(su[:8]),rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(native_t,tu[:8],rtol=2e-5,atol=2e-5)
        seed_all(seed+3001);model.mapping.reset_parameters()
        for p in model.mapping.parameters():p.requires_grad_(True)
        opt=torch.optim.Adam(model.mapping.parameters(),lr=cfg['emcdr_lr'],weight_decay=0.)
        gen=torch.Generator(device='cuda').manual_seed(seed+6100);history=[];chosen=None
        epochs=cfg['emcdr_epochs'] if phase=='select' else selections['EMCDR']['epoch']
        for epoch in range(1,epochs+1):
            total=torch.zeros((),device='cuda')
            for ix in torch.randperm(len(tid),device='cuda',generator=gen).split(cfg['emcdr_batch']):
                opt.zero_grad(set_to_none=True);loss=(model.mapping(su[ix])-tu[ix]).square().mean();loss.backward();opt.step();total+=loss.detach()*len(ix)
            row=dict(epoch=epoch,mapping_mse=total.item()/len(tid))
            if phase=='select':
                pred=em_predict(model,vx,seen,base);row['validation_rmse']=float(np.sqrt(np.square(pred-val[:,2]).mean()))
                if chosen is None or row['validation_rmse']<chosen['validation_rmse']:
                    chosen=dict(row);torch.save(model.mapping.state_dict(),d/'emcdr.pt');np.save(d/'emcdr_validation.npy',pred)
            history.append(row);save(d/'emcdr_history.json',history)
        if phase=='select':
            selections['EMCDR']=chosen;model.mapping.load_state_dict(torch.load(d/'emcdr.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(em_predict(model,vx,seen,base),np.load(d/'emcdr_validation.npy'))
        else:
            torch.save(model.mapping.state_dict(),d/'emcdr.pt');pred=em_predict(model,vx,seen,base)
            np.save(out/'EMCDR_test.npy',pred)
            model.mapping.load_state_dict(torch.load(d/'emcdr.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(em_predict(model,vx,seen,base),pred)
        for p in model.parameters():p.requires_grad_(False)
        inv_domain(d,model,name,'tgt',tids,cfg,seed)
        tpdata=np.load(d/'prototypes_tgt.npz');np.testing.assert_array_equal(tpdata['user_ids'],tids)
        tp=torch.tensor(tpdata['prototypes'],device='cuda')
        seed_all(seed+3000);model.rp_mapping.reset_parameters()
        for p in model.rp_mapping.parameters():p.requires_grad_(True)
        opt=torch.optim.Adam(model.rp_mapping.parameters(),lr=cfg['bridge_lr'],weight_decay=0.)
        gen=torch.Generator(device='cuda').manual_seed(seed+6000);history=[];chosen=None
        epochs=cfg['bridge_epochs'] if phase=='select' else selections['PFCDR']['epoch']
        eids=torch.tensor(eval_ids,device='cuda',dtype=torch.long);k=cfg['primary_k']
        for epoch in range(1,epochs+1):
            total=torch.zeros((),device='cuda')
            for ix in torch.randperm(len(tid),device='cuda',generator=gen).split(cfg['bridge_batch']):
                opt.zero_grad(set_to_none=True);loss=(model.rp_mapping(sp[tid[ix]])-tp[ix]).square().mean();loss.backward();opt.step();total+=loss.detach()*len(ix)
            row=dict(epoch=epoch,mapping_mse=total.item()/len(tid))
            if phase=='select':
                vectors=pf.voted_vectors(model,name,sp[eids],[k])[k]
                pred=pf.rating_predictions(model,name,vectors,eids,vx,base,seen)
                row['validation_rmse']=float(np.sqrt(np.square(pred-val[:,2]).mean()))
                if chosen is None or row['validation_rmse']<chosen['validation_rmse']:
                    chosen=dict(row);torch.save(model.rp_mapping.state_dict(),d/'pfcdr.pt');np.save(d/'pfcdr_validation.npy',pred)
            history.append(row);save(d/'pfcdr_history.json',history)
            print('BRIDGE',beta,seed,name,phase,json.dumps(row),flush=True)
        if phase=='select':
            selections['PFCDR']=chosen
            model.rp_mapping.load_state_dict(torch.load(d/'pfcdr.pt',map_location='cuda',weights_only=True))
            vectors=pf.voted_vectors(model,name,sp[eids],[k])[k]
            np.testing.assert_array_equal(pf.rating_predictions(model,name,vectors,eids,vx,base,seen),np.load(d/'pfcdr_validation.npy'))
            save(out/'selection.json',selections)
        else:
            torch.save(model.rp_mapping.state_dict(),d/'pfcdr.pt')
            vectors=pf.voted_vectors(model,name,sp[eids],[k])[k]
            pred=pf.rating_predictions(model,name,vectors,eids,vx,base,seen)
            np.save(out/'PFCDR_test.npy',pred);np.save(out/'test_vectors.npy',vectors.cpu().numpy())
            torch.save(model.state_dict(),out/'final.pt')
            restored=pf.make_model(name,m,10);restored.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True))
            restored.eval()
            np.testing.assert_array_equal(em_predict(restored,vx,seen,base),np.load(out/'EMCDR_test.npy'))
            saved_vectors=torch.tensor(np.load(out/'test_vectors.npy'),device='cuda')
            np.testing.assert_array_equal(pf.rating_predictions(restored,name,saved_vectors,eids,vx,base,seen),pred)
            # Native published voting on real checkpoint/prototypes, independent of cache.
            for j in [0,len(vx)//3,len(vx)//2,len(vx)-1]:
                pair=vx[j:j+1];native=restored(torch.cat([pair[:,1:2].float(),sp[pair[:,0]]],1),'test_source_free')
                idx=int(np.searchsorted(eval_ids,int(pair[0,0])))
                raw=(saved_vectors[idx]*pf.table(restored,name,'tgt').iid_embedding(pair[:,1])).sum(1)
                torch.testing.assert_close(native,raw,rtol=3e-5,atol=3e-5)
            for key,value in torch.load(src/'source.pt',weights_only=True,map_location='cuda').items():
                torch.testing.assert_close(restored.src_model.state_dict()[key],value,rtol=0,atol=0)
            assert (out/'selection.json').stat().st_mtime_ns<(out/'final.pt').stat().st_mtime_ns
            save(out/'complete.json',dict(status='PASS',seed=seed,beta=beta,backbone=name,
                selection_sha256=digest(out/'selection.json'),final_sha256=digest(out/'final.pt'),
                predictions={a:digest(out/f'{a}_test.npy') for a in ['EMCDR','PFCDR']},
                restored_predictions=True,native_voting_checks=4,source_weights_unchanged=True,
                full_outer_train_rows=len(data),test_labels_opened=False))
            del restored
        del model,opt,data
    print('BASELINES_FROZEN',beta,seed,name,flush=True)


class ResidualList(ListStudent):
    def __init__(self,lists,vocab,items,base,cfg):
        super().__init__(lists,vocab,items,0.,cfg)
        self.item_bias.weight.requires_grad_(False)
        self.register_buffer('base',base,persistent=False)
        nn.init.zeros_(self.encoder[-1].weight);nn.init.zeros_(self.encoder[-1].bias)
    def correction(self,u,i):
        rows=self.lists[u]
        h=self.source_items(rows,per_sample_weights=self.rank_weights.expand_as(rows))
        z=self.encoder(h)*(rows[:,:1]!=0)
        return (z*self.target_items(i)).sum(-1)
    def forward(self,u,i):return self.base[i]+self.correction(u,i)


@torch.no_grad()
def corrections(model,x,seen):
    model.eval();p=torch.cat([model.correction(b[:,0],b[:,1]) for b in x.split(4096)])
    return torch.where(seen[x[:,1]],p,0.).cpu().numpy()


def list_run(root,seed,beta,arm):
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
        seed_all(seed);model=ResidualList(lists,len(vocab),m['target_items'],base,cfg).cuda()
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
            restored=ResidualList(lists,len(vocab),m['target_items'],base,cfg).cuda()
            restored.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(restored,tx,seen),p)
            save(out/'complete.json',dict(status='PASS',selection_sha256=digest(out/'selection.json'),final_sha256=digest(out/'final.pt'),test_sha256=digest(out/'test.npy'),
                assignment_sha256=digest(out/'assignment.npy'),initial_parameters_sha256=init_hashes[0],
                full_outer_train_rows=len(data),refit_epochs=epochs,refit_reinitialized=True,restored_predictions=True,test_labels_opened=False))
            del restored
        del model,opt,x,y,data
    print('RESIDUAL_FROZEN',beta,seed,arm,flush=True)


def selftest(root):
    pf.BACKBONES=BACKBONES;pf.selftest(root)
    cfg=read(root/'protocol.json');seed_all(2020)
    lists=torch.tensor([[1,2],[2,1],[0,0]],device='cuda');base=torch.tensor([3.,4.,2.],device='cuda')
    model=ResidualList(lists,2,3,base,cfg).cuda();model.eval()
    u=torch.arange(3,device='cuda');i=u.clone()
    torch.testing.assert_close(model(u,i),base,rtol=0,atol=0)
    (model(u,i)-5).square().mean().backward()
    assert model.encoder[-1].weight.grad.abs().sum()>0
    with torch.no_grad():model.encoder[-1].bias.fill_(1.)
    assert model(u,i)[-1]==base[-1]
    save(root/'residual_selftest.json',dict(status='PASS',starts_at_ordinary_mean=True,zero_init_can_learn=True,no_list_stays_at_mean=True))


def evaluate(root):
    check_code(root);cfg=read(root/'protocol.json')
    assert read(root/'selftest.json')['status']==read(root/'residual_selftest.json')['status']=='PASS'
    frozen={}
    for beta in BETAS:
        for seed in SEEDS:
            for name in BACKBONES+['residual_'+a for a in ARMS]:
                p=root/f'beta_{beta}/seed_{seed}/{name}';c=read(p/'complete.json');assert c['status']=='PASS'
                assert digest(p/'selection.json')==c['selection_sha256'] and digest(p/'final.pt')==c['final_sha256']
                if name in BACKBONES:
                    for method,h in c['predictions'].items():assert digest(p/f'{method}_test.npy')==h
                else:assert digest(p/'test.npy')==c['test_sha256']
                frozen[str(p.relative_to(root))]=digest(p/'complete.json')
            a=root/f'beta_{beta}/seed_{seed}'
            assert read(a/'residual_real/complete.json')['initial_parameters_sha256']==read(a/'residual_shuffled/complete.json')['initial_parameters_sha256']
    with (root/'evaluation_started.json').open('x') as f:json.dump(frozen,f,indent=2)
    rows=[];seed_rows=[];results={}
    for beta in BETAS:
        cell=root/f'beta_{beta}';m=read(cell/'manifest.json')
        assert digest(cell/'data/test.csv')==m['hashes']['data/test.csv']
        test=frame(cell/'data/test.csv');np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
        y=test[:,2];_,users=np.unique(test[:,0],return_inverse=True)
        mean=predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean')
        tr=pd.read_csv(cell/'data/train.csv',header=None,names=['uid','iid','y'])
        independent=pd.Series(test[:,1].astype(int)).map(tr.groupby('iid').y.mean()).fillna(tr.y.mean()).to_numpy()
        np.testing.assert_allclose(mean,independent,rtol=0,atol=1e-12)
        predictions={}
        for name in BACKBONES:
            for method in ['EMCDR','PFCDR']:
                predictions[f'{method}_{name}']=np.stack([np.load(cell/f'seed_{s}/{name}/{method}_test.npy') for s in SEEDS])
        old=read(cell/'old_list.json')
        for s,v in old.items():assert digest(Path(v['path'])/'test.npy')==v['test_sha256']
        predictions['LIST_original']=np.stack([np.load(Path(old[str(s)]['path'])/'test.npy') for s in SEEDS])
        for arm in ARMS:predictions['LIST_residual_'+arm]=np.stack([np.load(cell/f'seed_{s}/residual_{arm}/test.npy') for s in SEEDS])
        baseline=metrics(mean,y,users);models={}
        rows.append(dict(beta=beta,method='item_mean',mae=baseline['mae'],mae_std=0.,rmse=baseline['rmse'],rmse_std=0.))
        for name,pred in predictions.items():
            per_seed={str(s):metrics(pred[i],y,users) for i,s in enumerate(SEEDS)}
            aggregate={key:dict(mean=float(np.mean([v[key] for v in per_seed.values()])),std=float(np.std([v[key] for v in per_seed.values()],ddof=1))) for key in ['mae','rmse']}
            comparisons={ref:paired_seed_mean(pred,value,y,users,cfg) for ref,value in {'item_mean':mean[None,:],'LIST_original':predictions['LIST_original']}.items()}
            if name=='LIST_residual_real':comparisons['shuffled']=paired_seed_mean(pred,predictions['LIST_residual_shuffled'],y,users,cfg)
            for v in comparisons.values():v['interpretation']='negative favors method; mean of three individual model metrics; user-cluster interval conditional on fitted models and fixed split; no multiple comparison correction'
            models[name]=dict(aggregate=aggregate,per_seed=per_seed,comparisons=comparisons)
            rows.append(dict(beta=beta,method=name,mae=aggregate['mae']['mean'],mae_std=aggregate['mae']['std'],rmse=aggregate['rmse']['mean'],rmse_std=aggregate['rmse']['std']))
            for s,v in per_seed.items():seed_rows.append(dict(beta=beta,method=name,seed=s,mae=v['mae'],rmse=v['rmse']))
        results[str(beta)]=dict(baseline=baseline,models=models,manifest=m)
    save(root/'summary.json',dict(status='PASS',protocol=cfg,results=results))
    for name,data in [('comparison',rows),('per_seed',seed_rows)]:
        with (root/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save(root/'verification.json',dict(status='PASS',baseline_pairs=12,residual_models=12,all_new_models_frozen_before_test=True,independent_means_match=True,real_shuffled_initial_parameters_match=True))
    lines=['# 最小论文任务 Music→Movie：LIST均值修正与跨域基线','',
        '去重后17,904名重叠用户；复用8638的20%和80%测试比例、内层验证与完整外层重训协议。每个方法三个种子。',
        '均值基线不平滑；LIST原版复用既有冻结预测。表中为各模型指标的算术均值±样本标准差，不是预测集成。','',
        '| 冷用户比例 | 方法 | MAE ± 标准差 | RMSE ± 标准差 |','| --- | --- | ---: | ---: |']
    for r in rows:lines.append(f"| {r['beta']}% | {r['method']} | {r['mae']:.6f} ± {r['mae_std']:.6f} | {r['rmse']:.6f} ± {r['rmse_std']:.6f} |")
    lines+=['','## 方法和公平性边界','',
        '源+目标总评分数最小的论文任务是Music→Movie（原始2,795,125条，去重2,760,244条）。Book→Music的重叠用户更少，但总评分9,995,633条，不是训练规模最小。',
        'MF/DNN的EMCDR与PFCDR使用相同骨干预训练权重、相同训练用户。DNN源模型与LIST源列表来自同一8638权重。MF为论文主表骨干，DNN为补充对照。',
        '源预训练batch512，保留既有LIST源模型；作者task3配置为256，因此不是逐参数、十次重复的论文复现。目标预训练batch256、dim10、10轮、Adam0.01。',
        'EMCDR使用作者无偏置映射，DNN映射有效用户向量，Adam0.01、batch128、最多10轮。PFCDR反演300步、batch200、目标评分5、Adam1e-4及原调度；原型映射Adam0.005、batch1024、最多10轮。',
        'PFCDR保留公开代码实际K70000，以及包含未训练冷用户行的目标候选池；没有K或学习率搜索，不代表调参后的最佳PFCDR。MF/DNN原生投票与缓存评分通过核验。',
        '所有方法在相同内层训练和验证用户上选轮数，再重新初始化目标模型并在全部外层目标训练评分上重训；测试用户不进入目标训练。最终训练物品均值相同。',
        '优化LIST=固定普通物品均值+lambda×列表修正；冻结物品偏置，编码器最后一层置零，从纯均值开始。修正结构、维度32、hidden64、dropout0.1、Adam0.001、weight_decay1e-5、batch2048保持不变。',
        '训练时lambda=1；验证在epoch0..10和lambda={0,0.25,0.5,0.75,1}中联合选择，然后固定轮数和lambda重训。允许退回纯均值，但不保证测试不劣于均值。未新增残差惩罚超参数。',
        '优化LIST的真实/打乱组按原inner_train、validation、test用户组分别固定错配列表，再各自完整重训；两组同初始化、同批次种子，保持只读列表ID/排名的黑盒接口。无源列表用户修正为零。',
        'EMCDR/PFCDR可访问源模型内部信息，LIST只接收Top100 ID/排名，接口条件不同。测试训练未见物品均回退普通均值。',
        '全部新模型选型、重训和恢复核验完成后才统一读取测试标签。既有测试已用于历史探索，本轮不是新盲测；不使用测试调参。区间按用户簇重采样，条件于三个拟合种子及固定划分，未校正多重比较。','']
    (root/'report.md').write_text('\n'.join(lines));print('ALL_COMPARISONS_PASS',json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','selftest','source','baseline','residual','evaluate'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--beta',type=int,choices=BETAS)
    p.add_argument('--backbone',choices=BACKBONES);p.add_argument('--arm',choices=ARMS)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='source':source(root,a.seed,a.backbone)
    elif a.stage=='baseline':baseline_run(root,a.seed,a.beta,a.backbone)
    elif a.stage=='residual':list_run(root,a.seed,a.beta,a.arm)
    else:globals()[a.stage](root)
