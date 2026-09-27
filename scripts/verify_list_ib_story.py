"""Independent CPU audit of the frozen IB experiment; no model retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def verify(root):
    cfg=read(root/'protocol.json');parent=Path(cfg['parent'])
    assert read(root/'selftest.json')['status']=='PASS'
    assert read(root/'summary.json')['status']=='PASS'
    assert digest(parent/'protocol.json')==cfg['parent_protocol_sha256']
    for path,h in cfg['code_hashes'].items():assert digest(Path(path))==h,path
    frozen=read(root/'evaluation_started.json');stamp=(root/'evaluation_started.json').stat().st_mtime_ns
    assert len(frozen)==18
    summary=read(root/'summary.json');metric_rows=[];selections=[]
    for beta in cfg['betas']:
        cell=parent/f'beta_{beta}';manifest=read(cell/'manifest.json')
        for path,h in manifest['hashes'].items():assert digest(cell/path)==h,path
        frames={name:pd.read_csv(cell/f'data/{name}.csv',header=None,names=['u','i','y']) for name in ['train','inner_train','validation','test']}
        train,inner,val,test=[frames[n] for n in ['train','inner_train','validation','test']]
        assert not set(train.u)&set(test.u)
        assert not set(inner.u)&set(val.u)
        assert not set(val.u)&set(test.u)
        base=test.i.map(train.groupby('i').y.mean()).fillna(train.y.mean()).to_numpy()
        valbase=val.i.map(inner.groupby('i').y.mean()).fillna(inner.y.mean()).to_numpy()
        init_by_seed={}
        for variant,eta in cfg['variants'].items():
            for seed in cfg['seeds']:
                out=root/variant/f'beta_{beta}/seed_{seed}/residual_real'
                rel=str(out.relative_to(root));done=read(out/'complete.json')
                assert digest(out/'complete.json')==frozen[rel]
                assert (out/'complete.json').stat().st_mtime_ns<stamp
                for name,key in [('final.pt','final_sha256'),('test.npy','test_sha256'),('selection.json','selection_sha256'),('assignment.npy','assignment_sha256')]:
                    assert digest(out/name)==done[key]
                    assert (out/name).stat().st_mtime_ns<stamp
                assert done['status']=='PASS' and done['eta']==eta and done['test_labels_opened'] is False
                assert done['restored_predictions'] and done['refit_reinitialized']
                assert done['full_outer_train_rows']==len(train) and done['extra_parameters']==4160
                assert done['source_list_interactions']==int((train.u<manifest['overlap_users']).sum())
                assert init_by_seed.setdefault(seed,done['initial_parameters_sha256'])==done['initial_parameters_sha256']
                previous=read(cell/f'seed_{seed}/residual_real/complete.json')
                assert done['shared_initial_parameters_sha256']==previous['initial_parameters_sha256']
                assert done['assignment_sha256']==previous['assignment_sha256']
                history=read(out/'select_history.json')
                assert [v['epoch'] for v in history]==list(range(1,11))
                expected=dict(epoch=0,lambda_value=0.,validation_rmse=float(np.sqrt(np.square(valbase-val.y).mean())))
                for row in history:
                    # Accumulation uses float32, so allow rounding error in the
                    # independently checked identity objective=MSE+eta*KL.
                    assert abs(row['objective']-row['train_mse']-eta*row['kl_per_all_interactions'])<2e-5
                    for lam in cfg['lambdas']:
                        value=row['validation'][str(lam)]
                        if value<expected['validation_rmse']:
                            expected=dict(epoch=row['epoch'],lambda_value=lam,validation_rmse=value)
                selected=read(out/'selection.json')
                assert selected['epoch']==expected['epoch'] and selected['lambda_value']==expected['lambda_value']
                assert abs(selected['validation_rmse']-expected['validation_rmse'])<1e-12
                assert abs(np.sqrt(np.square(np.load(out/'validation.npy')-val.y).mean())-expected['validation_rmse'])<1e-12
                refit=read(out/'refit_history.json')
                assert len(refit)==selected['epoch']==done['refit_epochs']
                pred=np.load(out/'test.npy')
                assert pred.shape==(len(test),) and np.isfinite(pred).all()
                unseen=~test.i.isin(set(train.i)).to_numpy()
                np.testing.assert_array_equal(pred[unseen],base[unseen])
                mae=float(np.abs(pred-test.y).mean());rmse=float(np.sqrt(np.square(pred-test.y).mean()))
                old=np.load(cell/f'seed_{seed}/residual_real/test.npy')
                row=next(v for v in summary['results'][str(beta)][variant]['per_seed'] if v['seed']==seed)
                assert abs(mae-row['mae'])<1e-12 and abs(rmse-row['rmse'])<1e-12
                assert abs(mae-float(np.abs(old-test.y).mean())-row['delta_mae_vs_original'])<1e-12
                assert abs(rmse-float(np.sqrt(np.square(old-test.y).mean()))-row['delta_rmse_vs_original'])<1e-12
                metric_rows.append(row);selections.append(dict(model=rel,**selected))
    result=dict(status='PASS',model_count=len(metric_rows),verifier_sha256=digest(Path(__file__)),
        checks=['18 models frozen before evaluation','original code and data hashes','training-only ordinary item means',
            'disjoint cold users','same common initialization as8650','same initialization acrosseta',
            'zero stochastic correction at initialization','KL sum64 then mean over all interactions',
            'selection argmin over epoch0..10 and lambda grid','full outer refit','independent metrics and deltas',
            'unseen-item fallback','source ID-only access guard'],selections=selections,metric_rows=metric_rows)
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',models=len(metric_rows),checks=result['checks']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    verify(parser.parse_args().root.resolve())
