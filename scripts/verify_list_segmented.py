"""Independent CPU audit of frozen segmented-LIST artifacts (no model retraining)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def verify(root):
    cfg=read(root/'protocol.json');parent=Path(cfg['parent'])
    assert read(root/'summary.json')['status']=='PASS'
    assert read(root/'selftest.json')['status']=='PASS'
    assert digest(parent/'protocol.json')==cfg['parent_protocol_sha256']
    for name,h in cfg['code_hashes'].items():assert digest(Path(name))==h,name
    frozen=read(root/'evaluation_started.json');evaluation_time=(root/'evaluation_started.json').stat().st_mtime_ns
    assert len(frozen)==24
    selections=[];metrics_checked=[]
    for beta in cfg['betas']:
        cell=parent/f'beta_{beta}'
        manifest=read(cell/'manifest.json')
        for name,h in manifest['hashes'].items():assert digest(cell/name)==h,name
        train=pd.read_csv(cell/'data/train.csv',header=None,names=['u','i','y'])
        inner=pd.read_csv(cell/'data/inner_train.csv',header=None,names=['u','i','y'])
        val=pd.read_csv(cell/'data/validation.csv',header=None,names=['u','i','y'])
        test=pd.read_csv(cell/'data/test.csv',header=None,names=['u','i','y'])
        assert not set(train.u)&set(test.u)
        assert not set(inner.u)&set(val.u)
        assert not set(val.u)&set(test.u)
        val_base=val.i.map(inner.groupby('i').y.mean()).fillna(inner.y.mean()).to_numpy()
        base=test.i.map(train.groupby('i').y.mean()).fillna(train.y.mean()).to_numpy()
        parent_result=read(parent/'summary.json')['results'][str(beta)]['baseline']
        assert abs(np.sqrt(np.square(base-test.y).mean())-parent_result['rmse'])<1e-12
        init_by_seed={};common_by_seed={}
        for variant in cfg['variants']:
            for seed in cfg['seeds']:
                for arm in cfg['arms']:
                    out=root/variant/f'beta_{beta}/seed_{seed}/residual_{arm}'
                    rel=str(out.relative_to(root));done=read(out/'complete.json')
                    assert digest(out/'complete.json')==frozen[rel]
                    for filename,key in [('final.pt','final_sha256'),('test.npy','test_sha256'),('selection.json','selection_sha256'),('assignment.npy','assignment_sha256')]:
                        assert digest(out/filename)==done[key]
                        assert (out/filename).stat().st_mtime_ns<evaluation_time
                    assert (out/'complete.json').stat().st_mtime_ns<evaluation_time
                    assert done['status']=='PASS' and done['restored_predictions'] and done['refit_reinitialized']
                    assert done['test_labels_opened'] is False
                    assert done['full_outer_train_rows']==len(train)
                    assert done['trainable_parameters']-done['parent_trainable_parameters']==4096
                    assert init_by_seed.setdefault(seed,done['initial_parameters_sha256'])==done['initial_parameters_sha256']
                    assert common_by_seed.setdefault(seed,done['shared_initial_parameters_sha256'])==done['shared_initial_parameters_sha256']
                    original=parent/f'beta_{beta}/seed_{seed}/residual_{arm}'
                    assert digest(out/'assignment.npy')==digest(original/'assignment.npy')
                    assignment=np.load(out/'assignment.npy')
                    for group in ['inner_train','validation','test']:
                        ids=np.load(cell/f'data/{group}_users.npy')
                        np.testing.assert_array_equal(np.sort(assignment[ids]),np.sort(ids))
                        assert np.all(assignment[ids]!=ids) if arm=='shuffled' else np.all(assignment[ids]==ids)
                    history=read(out/'select_history.json')
                    assert [v['epoch'] for v in history]==list(range(1,cfg['epochs']+1))
                    expected={'epoch':0,'lambda_value':0.,'validation_rmse':float(np.sqrt(np.square(val_base-val.y).mean()))}
                    for epoch in history:
                        for lam in cfg['lambdas']:
                            value=epoch['validation'][str(lam)]
                            if value<expected['validation_rmse']:
                                expected=dict(epoch=epoch['epoch'],lambda_value=lam,validation_rmse=value)
                    selected=read(out/'selection.json')
                    assert selected['epoch']==expected['epoch'] and selected['lambda_value']==expected['lambda_value']
                    assert abs(selected['validation_rmse']-expected['validation_rmse'])<1e-12
                    vp=np.load(out/'validation.npy')
                    assert abs(np.sqrt(np.square(vp-val.y).mean())-expected['validation_rmse'])<1e-12
                    refit=read(out/'refit_history.json')
                    assert len(refit)==selected['epoch']==done['refit_epochs']
                    assert [v['epoch'] for v in refit]==list(range(1,selected['epoch']+1))
                    pred=np.load(out/'test.npy');assert pred.shape==(len(test),) and np.isfinite(pred).all()
                    seen_items=set(train.i)
                    unseen=~test.i.isin(seen_items).to_numpy()
                    np.testing.assert_array_equal(pred[unseen],base[unseen])
                    result=read(root/'summary.json')['results'][str(beta)][variant+'_'+arm]['per_seed'][str(seed)]
                    mae=float(np.abs(pred-test.y).mean());rmse=float(np.sqrt(np.square(pred-test.y).mean()))
                    assert abs(mae-result['mae'])<1e-12 and abs(rmse-result['rmse'])<1e-12
                    selections.append(dict(model=rel,epoch=selected['epoch'],lambda_value=selected['lambda_value'],trainable_parameters=done['trainable_parameters']))
                    metrics_checked.append(rel)
    result=dict(status='PASS',model_count=len(metrics_checked),verifier_sha256=digest(Path(__file__)),
        checks=['frozen code and data hashes','training-only ordinary means','disjoint train/validation/test users',
                'all 24 final models predate test evaluation','independent metric recomputation',
                'validation argmin over declared epochs and lambdas','full outer refit for selected epochs',
                'same initial parameters across new variants and arms','4096 extra parameters versus parent',
                'same within-group user derangements as parent','unseen-item fallback'],selections=selections)
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],models=result['model_count'],checks=result['checks']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    verify(p.parse_args().root.resolve())
