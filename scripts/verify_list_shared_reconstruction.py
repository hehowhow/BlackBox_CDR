"""Recompute six frozen seed2021 results and audit the declared new objective."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    cfg=read(root/'protocol.json')
    for path,expected in cfg['code_hashes'].items():
        assert sha(Path(path))==expected,path
    assert read(root/'selftest_cpu.json')['status']=='PASS'
    assert read(root/'verification.json')['status']=='PASS'
    assert read(root/'training_loop_equivalence.json')['status']=='PASS'
    assert cfg['seeds']==[2021] and cfg['betas']==[20,80]
    tasks=read(root/'tasks.json'); refs=read(root/'references.json')
    evaluation=read(root/'evaluation_started.json'); frozen=evaluation['frozen_models']
    initial=read(root/'initialization.json'); access=read(root/'training_list_access.json')
    assert len(tasks)==len(frozen)==6
    with (root/'results.csv').open() as handle: metrics=list(csv.DictReader(handle))
    with (root/'paired.csv').open() as handle: paired=list(csv.DictReader(handle))
    assert len(metrics)==24 and len(paired)==6
    maximum_error=0.; receipts=[]
    for request in tasks:
        task,beta,seed=request['task'],request['beta'],request['seed']
        assert seed==2021
        key=f'{task}/{beta}/{seed}'
        relative=Path('models')/task/f'beta_{beta}'/f'seed_{seed}'/'raw_energy'
        out=root/relative; done=read(out/'complete.json')
        assert sha(out/'complete.json')==frozen[str(relative)]
        assert done['status']=='PASS' and not done['test_labels_opened']
        assert done['finished'] <= evaluation['time']
        assert done['restored_predictions'] and done['refit_reinitialized']
        for file,field in [('final.pt','final_sha256'),('test.npy','test_sha256'),
                           ('selection.json','selection_sha256'),('encoder_audit.json','encoder_audit_sha256')]:
            assert sha(out/file)==done[field]
        audit=read(out/'encoder_audit.json'); trace=read(out/'objective_trace.json')
        assert audit['no_frequency_statistics'] and audit['train_only_masked_reconstruction']
        assert sha(out/'objective_trace.json')==audit['objective_trace_sha256']
        receipt=read(out/'started.json');receipts.append(receipt['gpu'])
        assert receipt['execution_backend']=='direct' and receipt['job'] is None
        assert receipt['authorization_sha256']==sha(root/'direct_authorization.json')
        assert read(out/'access_audit.json')['status']=='PASS'
        selection=read(out/'selection.json'); history=read(out/'select_history.json')
        assert len(history)==cfg['epochs']==10
        choices=[(history[0]['validation']['0.0'],0,0.)]
        choices += [(r['validation'][str(lam)],r['epoch'],lam) for r in history for lam in cfg['lambdas']]
        best=min(choices,key=lambda x:x[0])
        assert (selection['validation_rmse'],selection['epoch'],selection['lambda_value'])==best
        baseline=Path(refs[key]['LIST_langevin']['path']); old=read(baseline/'complete.json')
        for phase in ['select','refit']:
            meta=done['phases'][phase]; records=read(out/f'{phase}_history.json')
            assert len(records)==meta['epochs']
            assert all(r['updates']==math.ceil(meta['grouped_users']/64) for r in records)
            assert meta['initial_sha256']==initial[f'{key}/{phase}']['new_initial_sha256']
            assert initial[f'{key}/{phase}']['inherited_initial_sha256']==old['phases'][phase]['initial_sha256']
            assert meta['trainable_parameters']-old['phases'][phase]['trainable_parameters']==11584
            assert meta['fit_users_sha256']==access[f'{key}/{phase}']['fit_users_sha256']
            phase_trace=[r for r in trace if r['phase']==phase]
            assert len(phase_trace)==sum(r['updates'] for r in records)
            position=0
            for epoch in records:
                updates=phase_trace[position:position+epoch['updates']];position+=epoch['updates']
                expected=sum(r['total_objective']*r['ratings_count'] for r in updates)/meta['overlap_train_rows']
                assert abs(epoch['objective']-expected)<1e-12
            for row in phase_trace:
                expected=row['recommendation_objective']+cfg['shared_reconstruction']['reconstruction_weight']*row['reconstruction_normalized']
                assert abs(row['total_objective']-expected)<1e-5
                expected=row['reconstruction_ce']/math.log(access[f'{key}/{phase}']['candidate_count'])
                assert abs(row['reconstruction_normalized']-expected)<1e-6
                assert row['masked_positions']%20==0 and 20<=row['masked_positions']<=1280
        assert done['phases']['refit']['epochs']==selection['epoch']
        cell=root/'data'/task/f'beta_{beta}';manifest=read(cell/'manifest.json')
        assert sha(cell/'data/test.csv')==manifest['hashes']['data/test.csv']
        data=np.loadtxt(cell/'data/test.csv',delimiter=',');y=data[:,2]
        np.testing.assert_array_equal(data[:,:2],np.load(cell/'data/test_features.npy'))
        # Import the established ordinary mean predictor, independent of the new model.
        from book_movie_protocol import predict
        mean=predict(dict(np.load(cell/'baseline/statistics.npz')),data[:,1].astype(int),'item_mean')
        candidates=[('item_mean',None),('LIST_langevin',baseline),
                    ('LIST_frequency_residual_langevin',Path(refs[key]['LIST_frequency_residual_langevin']['path'])),
                    ('LIST_shared_reconstruction_langevin',out)]
        for name,path in candidates:
            if path is not None and name!='LIST_shared_reconstruction_langevin':
                ref=refs[key][name]
                assert sha(path/'complete.json')==ref['complete_sha256'] and sha(path/'test.npy')==ref['test_sha256']
            pred=mean if path is None else np.load(path/'test.npy')
            assert pred.shape==y.shape and np.isfinite(pred).all()
            row=next(r for r in metrics if r['task']==task and int(r['beta'])==beta and r['method']==name)
            errors=pred-y
            expected={'mae':float(np.abs(errors).mean()),'rmse':float(np.sqrt(np.square(errors).mean()))}
            for name,value in expected.items():
                difference=abs(float(row[name])-value);maximum_error=max(maximum_error,difference)
                assert difference<1e-12
            if float(row['lambda_value'])==0:
                np.testing.assert_array_equal(pred,mean)
    assert len(set(receipts))==2
    for row in paired:
        values={r['method']:r for r in metrics if all(r[k]==row[k] for k in ['task','beta','seed'])}
        for m in ['mae','rmse']:
            new=float(values['LIST_shared_reconstruction_langevin'][m])
            assert abs(float(row['delta_'+m])-(new-float(values['LIST_langevin'][m])))<1e-12
            assert abs(float(row['delta_vs_frequency_'+m])-(new-float(values['LIST_frequency_residual_langevin'][m])))<1e-12
    result=dict(status='PASS',models=6,metric_rows=24,seed=2021,maximum_metric_recomputation_error=maximum_error,
                validation_selection_reproduced=True,unchanged_training_budgets=True,
                declared_auxiliary_objective_verified=True,added_trainable_parameters=11584,
                baseline_prediction_hashes_verified=True,all_predictions_frozen_before_test=True,
                direct_execution_receipts=True,gpu_uuids=sorted(set(receipts)))
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    status_path=root/'execution_status.json'
    status=read(status_path) if status_path.exists() else {}
    status.update(status='COMPLETE',completed_models=6,finished=max(read(root/p/'complete.json')['finished'] for p in frozen),
                  verified_at=__import__('time').time(),independent_verification='PASS',results_available=True)
    status_path.write_text(json.dumps(status,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.root),indent=2))
