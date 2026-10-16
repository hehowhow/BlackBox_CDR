"""Independent protocol, selection, provenance and metric audit for the MF run."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(root):
    assert read(root/'verification.json')['status']=='PASS'
    cfg=read(root/'protocol.json')
    assert cfg['backbones']==['MF'] and cfg['seeds']==[2020] and cfg['betas']==[20,50,80]
    assert sha(root/'protocol.json')==read(root/'plan_frozen.json')['protocol_sha256']
    assert sha(root/'tasks.json')==read(root/'plan_frozen.json')['tasks_sha256']
    for p,h in cfg['code_hashes'].items():assert sha(Path(p))==h,p
    with (root/'results.csv').open() as f:results=list(csv.DictReader(f))
    assert len(results)==54
    index={(r['task'],int(r['beta']),r['method']):r for r in results}
    assert len(index)==54
    frozen=read(root/'all_predictions_frozen.json')['predictions']
    counts=[];max_error=0.
    for task,n in cfg['tasks'].items():
        tr=root/'baselines'/task
        tc=read(tr/'protocol.json')
        assert tc['source_batch']==tc['target_batch']==(256 if task=='music_movie_clean' else 512)
        assert tc['bridge_batch']==(1024 if task=='music_movie_clean' else 512)
        assert tc['pretrain_epochs']==tc['bridge_epochs']==tc['emcdr_epochs']==10
        assert tc['inversion_iterations']==300 and tc['primary_k']==70000
        src=tr/'sources/seed_2020/MF'
        assert len(read(src/'pretrain_src_history.json'))==10
        assert sum(x['users'] for x in read(src/'inversion_src_history.json'))==n
        query=root/'list_queries'/task/'seed_2020/queries/MF'
        qr=read(query/'complete.json')
        assert qr['backbone']=='MF' and qr['source_sha256']==sha(src/'source.pt')
        assert qr['item_ids_sha256']==sha(query/'topk_ids.npz')
        original=np.load(query/'topk_ids.npz')['item_ids']
        tokens=np.load(root/'published'/task/'raw.npz')['item_ids']
        assert original.shape==tokens.shape==(n,100)
        np.testing.assert_array_equal(np.unique(original)[tokens-1],original)
        for beta in [20,50,80]:
            cell=tr/f'beta_{beta}';m=read(cell/'manifest.json')
            assert m['overlap_users']==n and m['cross_domain_raw_item_overlap']==0
            users={name:np.load(cell/f'data/{name}_users.npy') for name in ['train','inner_train','validation','test']}
            assert len(users['test'])==round(n*beta/100)
            assert not np.intersect1d(users['train'],users['test']).size
            assert not np.intersect1d(users['inner_train'],users['validation']).size
            np.testing.assert_array_equal(np.union1d(users['inner_train'],users['validation']),users['train'])
            out=cell/'seed_2020/MF';done=read(out/'complete.json');sel=read(out/'selection.json')
            assert done['restored_predictions'] and done['native_voting_checks']==4
            assert done['source_weights_unchanged'] and done['raw_official_predictions_no_mean_fallback']
            assert done['tgtonly_epochs']==10 and done['tgtonly_same_target_checkpoint']
            for phase,which in [('select','inner_train'),('refit','train')]:
                assert len(read(out/phase/'pretrain_tgt_history.json'))==10
                np.testing.assert_array_equal(np.load(out/phase/'prototypes_tgt.npz')['user_ids'],users[which])
            for method in ['EMCDR','PFCDR']:
                history=read(out/'select'/f'{method.lower()}_history.json')
                assert len(history)==10
                chosen=min(history,key=lambda r:r['validation_rmse'])
                assert chosen['epoch']==sel[method]['epoch']
                assert len(read(out/'refit'/f'{method.lower()}_history.json'))==chosen['epoch']
            outputs={'LIST_V1':cell/'seed_2020/residual_real'}
            for method in ['LIST_V1_Langevin','LAPT']:
                outputs[method]=root/method/'models'/task/f'beta_{beta}/seed_2020/raw_energy'
            np.testing.assert_array_equal(np.load(outputs['LIST_V1']/'assignment.npy'),np.arange(n))
            initialization={}
            for method,p in outputs.items():
                done=read(p/'complete.json');choice=read(p/'selection.json')
                assert done['restored_predictions'] and done['refit_reinitialized']
                assert not done['test_labels_opened']
                assert sha(p/'selection.json')==done['selection_sha256']
                history=read(p/'select_history.json');refit=read(p/'refit_history.json')
                assert len(history)==10 and len(refit)==choice['epoch']
                candidates=[(float(v),r['epoch'],float(lam)) for r in history for lam,v in r['validation'].items()]
                if choice['epoch']==0:
                    assert choice['lambda_value']==0 and choice['validation_rmse']<=min(x[0] for x in candidates)+1e-12
                else:
                    expected=min(candidates,key=lambda x:x[0])
                    assert (choice['epoch'],choice['lambda_value'])==expected[1:]
                    assert abs(choice['validation_rmse']-expected[0])<1e-12
                if method!='LIST_V1':
                    initialization[method]=read(p/'initialization_alignment.json')
                    for phase,which in [('select','inner_train'),('refit','train')]:
                        release=root/'releases'/task/f'beta_{beta}/seed_2020'/phase
                        meta=read(release/'release.json')
                        assert meta['source_sha256']==qr['source_sha256']
                        assert done['phases'][phase]['release_sha256']==sha(release/'raw.npz')
                        assert done['phases'][phase]['grouped_users']==len(users[which])
                        assert done['phases'][phase]['fit_users_sha256']==hashlib.sha256(users[which].tobytes()).hexdigest()
            for a,b in zip(initialization['LIST_V1_Langevin'],initialization['LAPT']):
                assert a['phase']==b['phase'] and a['raw_initial_sha256']==b['raw_initial_sha256']
                assert b['extra_trainable_parameters']==11584
            data=np.loadtxt(cell/'data/test.csv',delimiter=',',ndmin=2)
            np.testing.assert_array_equal(data[:,:2],np.load(cell/'data/test_features.npy'))
            assert sha(cell/'data/test.csv')==m['hashes']['data/test.csv']
            for method in cfg['methods']:
                row=index[task,beta,method];p=Path(row['prediction']);values=np.load(p)
                assert sha(p)==frozen[f'{task}/{beta}/{method}']['sha256']
                assert values.shape==data[:,2].shape
                errors=values.astype(float)-data[:,2]
                mae=math.fsum(np.abs(errors))/len(errors)
                rmse=math.sqrt(math.fsum(errors*errors)/len(errors))
                error=max(abs(mae-float(row['mae'])),abs(rmse-float(row['rmse'])))
                assert error<1e-12
                max_error=max(max_error,error)
            counts.append(dict(task=task,beta=beta,train_users=len(users['train']),test_users=len(users['test']),test_ratings=len(data)))
    report=dict(status='PASS',results=54,source_backbone='MF',same_source_checkpoint_and_ranked_IDs=True,
        author_task_specific_batches=True,all_stage_epoch_counts_checked=True,
        validation_choices_recomputed=True,all_refits_reinitialized=True,
        no_target_train_test_user_overlap=True,matched_energy_initialization=True,
        lapt_extra_parameters=11584,metric_max_difference=max_error,cells=counts,
        verifier_sha256=sha(Path(__file__)))
    (root/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    verify(p.parse_args().root)
