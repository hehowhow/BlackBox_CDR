"""Validation-motivated GMF matching correction, reusing frozen base/prototypes."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import pfcdr_backbone_study as study
from book_movie_protocol import BASE, digest, save, predict
from residual_experiment import guard_test_labels, seed_all
from residual_summary import metrics, paired_interval

LABEL='GMF_score_matching'


def prepare(root):
    out=root/LABEL
    out.mkdir(exist_ok=False)
    cfg=study.read(root/'protocol.json')
    save(out/'control_protocol.json',dict(
        motivation='Poor validation performance under public GMF matcher; specified before main study test evaluation',
        difference='Match users with the actual GMF score linear(U * mapped_prototype), including learned dimension weights',
        reused='Identical source/target model, source/target prototypes, initial mapping, training users and batch sequence',
        seed=cfg['seed'],ks=cfg['reported_ks'],selection='validation RMSE separately for each K',
        extra_pretraining=False,extra_inversion=False,bridge_epochs=cfg['bridge_epochs'],
        script_sha256=digest(Path(__file__)),parent_protocol_sha256=digest(root/'protocol.json'),
        original_complete_sha256=digest(root/'GMF/complete.json'),
        main_test_evaluation_already_started=(root/'evaluation_started.json').exists()))
    assert not (root/'evaluation_started.json').exists()
    for f in ['prototypes_src.npz','prototypes_tgt.npz']:
        (out/f).symlink_to(root/'GMF'/f)
    print('GMF_MATCHING_CONTROL_FROZEN',flush=True)


@torch.no_grad()
def corrected_vectors(model,name,prototypes,ks,batch=32):
    assert name=='GMF'
    eff=study.effective_users(model,name,'tgt')
    result={k:[] for k in ks}
    for p in model.rp_mapping(prototypes).split(batch):
        order=torch.sort((eff@p.T-5).abs(),dim=0).indices
        for k in ks: result[k].append(eff[order[:k].T].mean(1))
    return {k:torch.cat(v) for k,v in result.items()}


def run(root):
    out=root/LABEL
    frozen=study.read(out/'control_protocol.json')
    assert digest(Path(__file__))==frozen['script_sha256']
    assert digest(root/'protocol.json')==frozen['parent_protocol_sha256']
    assert digest(root/'GMF/complete.json')==frozen['original_complete_sha256']
    cfg,manifest=study.read(root/'protocol.json'),study.read(root/'manifest.json')
    for p,h in {**cfg['code_sha256'],**cfg['upstream_sha256']}.items(): assert digest(Path(p))==h
    guard_test_labels(root)
    seed_all(cfg['seed'])
    model=study.make_model('GMF',manifest,cfg['dim'])
    model.load_state_dict(torch.load(root/'GMF/pretrained.pt',map_location='cuda',weights_only=True))
    # Change only this process's vote matcher; frozen original study is untouched.
    study.voted_vectors=corrected_vectors
    study.bridge(root,out,model,'GMF',cfg)
    val=pd.read_csv(root/'data/validation.csv',header=None).to_numpy()
    src=np.load(out/'prototypes_src.npz')['prototypes']
    rows=np.random.default_rng(923).choice(len(val),size=16,replace=False)
    checks=[]
    base,seen=np.load(root/'baseline/values.npy'),np.load(root/'baseline/seen.npy')
    for k,sel in study.read(out/'selection.json').items():
        model.rp_mapping.load_state_dict(torch.load(out/f'bridge_k{k}.pt',map_location='cuda',weights_only=True))
        reference=[]
        with torch.no_grad():
            raw=model.tgt_model.embedding.uid_embedding.weight
            for row in rows:
                uid,iid=val[row,:2].astype(int)
                mapped=model.rp_mapping(torch.tensor(src[uid],device='cuda'))
                # Independent scoring via native GMF layer, without factorizing it.
                scores=model.tgt_model.linear(raw*mapped).flatten()
                chosen=torch.argsort((scores-5).abs())[:int(k)]
                item=model.tgt_model.embedding.iid_embedding.weight[iid]
                p=model.tgt_model.linear(raw[chosen]*item).mean().item()
                reference.append(p if seen[iid] else float(base[iid]))
        stored=np.load(out/f'validation_k{k}.npy')[rows]
        np.testing.assert_allclose(reference,stored,rtol=5e-5,atol=5e-5)
        checks.append(dict(k=int(k),native_score_samples=len(rows),max_error=float(np.max(np.abs(np.array(reference)-stored)))))
    save(out/'verification.json',dict(status='PASS',checks=checks))
    save(out/'complete.json',dict(status='PASS',selection_sha256=digest(out/'selection.json')))
    print('GMF_CORRECTED_MATCHING_VERIFIED',flush=True)


def evaluate(root):
    out=root/LABEL
    assert study.read(out/'verification.json')['status']=='PASS'
    cfg=study.read(root/'protocol.json')
    selected=study.read(out/'selection.json')
    assert digest(out/'selection.json')==study.read(out/'complete.json')['selection_sha256']
    with (out/'evaluation_started.json').open('x') as f:json.dump(selected,f)
    test=pd.read_csv(root/'data/test.csv',header=None).to_numpy()
    y=test[:,2];_,users=np.unique(test[:,0],return_inverse=True)
    stats=dict(np.load(root/'baseline/statistics.npz'))
    result={}
    for k,sel in selected.items():
        p=np.load(out/f'test_k{k}.npy')
        assert digest(out/f'test_k{k}.npy')==sel['test_sha256']
        refs={n:np.load(root/n/f'test_k{k}.npy') for n in ['MF','GMF','DNN']}
        refs.update({n:predict(stats,test[:,1].astype(int),n) for n in ['item_mean','smoothed_mean_5','item_median']})
        ci={n:paired_interval(p[None,:],q[None,:],y,users,cfg) for n,q in refs.items()}
        for c in ci.values():c['interpretation']='Negative favors corrected GMF; conditional on one fitted seed'
        result[k]=dict(selection=sel,metrics=metrics(p,y,users),comparisons=ci)
    save(out/'summary.json',dict(status='PASS',results=result,scope='Validation-motivated correction; not an original published GMF result'))
    print('GMF_MATCHING_CONTROL_PASS',json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','evaluate']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.root.resolve())
