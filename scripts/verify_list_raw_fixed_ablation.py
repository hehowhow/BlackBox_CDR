"""Independent post-freeze verification of the paper-parameter raw ablation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(root):
    cfg=read(root/'protocol.json');parent=Path(cfg['parent']);gparent=Path(cfg['gaussian_parent'])
    assert read(root/'verification.json')['status']=='PASS';assert cfg['policy']=='fixed_ablation'
    plan=read(root/'plan_frozen.json');frozen=read(root/'evaluation_started.json')
    for n,k in [('references.json','references_sha256'),('tasks.json','tasks_sha256'),('initialization_audit.json','initialization_sha256')]:assert sha(root/n)==plan[k]
    keys=['k','dim','hidden','dropout','epochs','lr','weight_decay','user_batch_size','eval_user_batch','latent_std',
        'logvar_bounds','observation_variance','lambdas','train_steps','eval_steps','train_mala_step','eval_mala_step','train_chains','eval_chains','energy_bound']
    old_cfg=read(parent/'protocol.json')
    for k in keys:assert cfg[k]==old_cfg[k],k
    assert cfg['latent_std']==.5 and cfg['user_batch_size']==64
    for p,h in cfg['code_hashes'].items():assert sha(p)==h,p
    init=read(root/'initialization_audit.json');assert init['status']=='PASS'
    jobs=read(root/'tasks.json');assert len(jobs)==18
    for job in jobs:
        rel=Path('models')/job['task']/f"beta_{job['beta']}"/f"seed_{job['seed']}"/'raw_energy'
        out=root/rel;done=read(out/'complete.json')
        assert sha(out/'complete.json')==frozen['models'][str(rel)]
        assert done['non_source_initialization_matches_coarse'] and done['restored_predictions'] and done['refit_reinitialized']
        assert not done['test_labels_opened'] and done['finished']<=frozen['time'] and done['finished']>=plan['time']
        for f,k in [('final.pt','final_sha256'),('test.npy','test_sha256'),('selection.json','selection_sha256'),('sampling_audit.json','sampling_audit_sha256')]:assert sha(out/f)==done[k]
        counterpart=read(parent/rel.parent/'coarse_energy/complete.json')
        legacy=read(parent/rel/'complete.json')
        sampling=read(out/'sampling_audit.json');records=sampling['initialization_checks'];assert len(records)==2
        assert all(v==cfg[k] for k,v in sampling['controls'].items())
        expected=init['records'][f"{job['task']}/{job['seed']}/energy"]
        assert expected['raw_shared_sha256']==expected['coarse_shared_sha256']
        assert set(expected['original_differing_trainable_parameters'])=={'energy_net.0.weight','energy_net.0.bias'}
        for phase in ['select','refit']:
            rec=next(r for r in records if r['phase']==phase);assert {k:v for k,v in rec.items() if k!='phase'}==expected
            p=done['phases'][phase];c=counterpart['phases'][phase]
            assert rec['coarse_initial_sha256']==c['initial_sha256']
            assert rec['legacy_raw_initial_sha256']==legacy['phases'][phase]['initial_sha256']
            assert p['initial_sha256']==rec['raw_initial_sha256']
            for k in ['common_initial_sha256','overlap_train_rows','grouped_users','full_target_train_rows','fit_users_sha256']:assert p[k]==c[k]
            assert p['trainable_parameters']-c['trainable_parameters']==32*(p['vocab_size']-c['vocab_size'])
            history=read(out/f'{phase}_history.json');assert len(history)==p['epochs']
            updates=math.ceil(p['grouped_users']/64);assert all(r['updates']==updates for r in history)
            assert sampling['acceptance'][phase]['batches']==len(history)*updates
            release=root/'releases'/job['task']/f"beta_{job['beta']}"/f"seed_{job['seed']}"/phase
            assert sha(release/'raw.npz')==p['release_sha256']==legacy['phases'][phase]['release_sha256']
        assert done['phases']['select']['initial_sha256']==done['phases']['refit']['initial_sha256']
        for f in read(out/'access_audit.json')['opened']:
            p=Path(f);assert not {'trusted','queries','id_maps','tiger_sid_content'}.intersection(p.parts)
            assert p.name not in {'coarse.npz','sid.npz','test.csv','test_labels.npy','source.pt','pretrained.pt','scores.npy'}
        history=read(out/'select_history.json');selected=read(out/'selection.json');assert len(history)==10
        best=(history[0]['validation']['0.0'],0,0.)
        for r in history:
            for lam in cfg['lambdas']:
                score=r['validation'][str(lam)]
                if score<best[0]:best=(score,r['epoch'],lam)
        assert best==(selected['validation_rmse'],selected['epoch'],selected['lambda_value'])
    refs=read(root/'references.json')
    # Independent CPU reconstruction of the Gaussian reused trainable tensors.
    import torch
    from residual_experiment import seed_all
    from small_task_comparison import ResidualList
    from list_coarse_sid_model import make_model
    from list_ambiguity_story import parameter_hash
    torch.set_num_threads(2)
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            meta=read(root/'releases'/task/'beta_20'/f'seed_{seed}'/'select/release.json')
            manifest=read(root/'data'/task/'beta_20/manifest.json')
            tokens=torch.zeros((1,100),dtype=torch.long);base=torch.zeros(manifest['target_items'])
            seed_all(seed);original=ResidualList(tokens,meta['raw_vocab_size'],manifest['target_items'],base,cfg)
            common={n:p.detach().clone() for n,p in original.named_parameters() if not n.startswith('source_items.')};del original
            models={}
            for kind in ['raw','coarse']:
                seed_all(seed);m=make_model(tokens,meta[kind+'_vocab_size'],manifest['target_items'],base,cfg,'gaussian')
                with torch.no_grad():
                    for n,p in m.named_parameters():
                        if n in common:p.copy_(common[n])
                assert parameter_hash(m)==init['records'][f'{task}/{seed}/gaussian'][('legacy_raw' if kind=='raw' else 'coarse')+'_initial_sha256']
                models[kind]=m
            cp=dict(models['coarse'].named_parameters())
            for n,p in models['raw'].named_parameters():
                if p.requires_grad and not n.startswith('source_items.'):assert torch.equal(p,cp[n]),n
            del models,common,cp,m
    rows=list(csv.DictReader((root/'per_seed.csv').open()));aggregates=list(csv.DictReader((root/'results.csv').open()))
    paired=list(csv.DictReader((root/'paired.csv').open()))
    assert len(rows)==144 and len(aggregates)==48 and len(paired)==36
    errors=[]
    for task in cfg['tasks']:
        for beta in cfg['betas']:
            cell=root/'data'/task/f'beta_{beta}';manifest=read(cell/'manifest.json')
            assert sha(cell/'data/test.csv')==manifest['hashes']['data/test.csv']
            test=np.loadtxt(cell/'data/test.csv',delimiter=',');np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
            stats=np.load(cell/'baseline/statistics.npz')
            meanmap={int(i):float(s)/int(c) for i,s,c in zip(stats['ids'],stats['sums'],stats['counts']) if c>0}
            ordinary=np.asarray([meanmap.get(int(i),float(stats['global_mean'])) for i in test[:,1]])
            for row in [r for r in rows if r['task']==task and int(r['beta'])==beta]:
                name=row['method'];seed=int(row['seed'])
                if name=='item_mean':pred=ordinary
                else:
                    path=Path(row['source_path']);pred=np.load(path/'test.npy').astype(np.float64)
                    if name!='raw_energy':
                        ref=refs[f'{task}/{beta}/{seed}'][name];assert str(path)==ref['path']
                        assert sha(path/'complete.json')==ref['complete_sha256'] and sha(path/'test.npy')==ref['test_sha256']
                delta=pred-test[:,2];metrics=dict(mae=float(np.abs(delta).mean()),rmse=float(np.sqrt(np.square(delta).mean())))
                errors.extend(abs(metrics[k]-float(row[k])) for k in metrics)
    for r in aggregates:
        chosen=[a for a in rows if all(a[k]==r[k] for k in ['task','beta','method'])];assert len(chosen)==3
        for k in ['mae','rmse']:
            a=[float(x[k]) for x in chosen];errors.extend([abs(np.mean(a)-float(r[k])),abs(np.std(a,ddof=1)-float(r[k+'_std']))])
    for r in paired:
        vals={a['method']:a for a in rows if all(a[k]==r[k] for k in ['task','beta','seed'])}
        for k in ['mae','rmse']:errors.append(abs(float(r['delta_'+k])-(float(vals['raw_'+r['family']][k])-float(vals['coarse_'+r['family']][k]))))
    assert max(errors)<1e-12
    result=dict(status='PASS',new_raw_energy_models=18,reused_raw_gaussian_models=18,per_seed_rows=144,aggregate_rows=48,paired_rows=36,
        paper_parameters_unchanged=True,raw_energy_shared_initialization_matches_coarse=True,reused_gaussian_active_initialization_matches_coarse=True,
        old_artifacts_unchanged=True,max_metric_error=max(errors),verifier_sha256=sha(__file__))
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);verify(p.parse_args().root)
