"""Independent CPU checks of cluster-count search, final metrics and training-only groups."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(root):
    cfg=read(root/'protocol.json');parent=Path(cfg['parent']);prior=Path(cfg['cluster_parent'])
    frozen=read(root/'evaluation_started.json');evaluation_time=(root/'evaluation_started.json').stat().st_mtime_ns
    assert read(root/'verification.json')['status']=='PASS'
    for p,h in cfg['code_hashes'].items():assert digest(Path(p))==h,p
    api={}
    for seed in cfg['seeds']:
        path=parent.parent/f'paper_list_dnn_8638/music_movie_clean/seed_{seed}/queries/DNN/topk_ids.npz'
        ids=np.load(path)['item_ids'];vocab=np.unique(ids);api[seed]=np.searchsorted(vocab,ids)+1
    cluster_checks=[];selection_checks=[];metric_checks=[]
    for beta in cfg['betas']:
        cell=parent/f'beta_{beta}';choice=read(root/f'choice_beta_{beta}.json')
        assert digest(root/f'choice_beta_{beta}.json')==frozen['choices'][str(beta)]
        inner=pd.read_csv(cell/'data/inner_train.csv',header=None,names=['u','i','y'])
        train=pd.read_csv(cell/'data/train.csv',header=None,names=['u','i','y'])
        val=pd.read_csv(cell/'data/validation.csv',header=None,names=['u','i','y'])
        test=pd.read_csv(cell/'data/test.csv',header=None,names=['u','i','y'])
        assert not set(inner.u)&set(val.u) and not set(train.u)&set(test.u)
        baseline=val.i.map(inner.groupby('i').y.mean()).fillna(inner.y.mean()).to_numpy()
        basevalrmse=float(np.sqrt(np.square(baseline-val.y).mean()))
        seen_selections=set()
        def check_selection(out):
            if out in seen_selections:return read(out/'selection.json')['validation_rmse']
            seen_selections.add(out)
            history=read(out/'select_history.json');expected=(basevalrmse,0,0.)
            assert [v['epoch'] for v in history]==list(range(1,cfg['epochs']+1))
            for v in history:
                for lam in cfg['lambdas']:
                    value=v['validation'][str(lam)]
                    if value<expected[0]:expected=(value,v['epoch'],lam)
            sel=read(out/'selection.json');done=read(out/'select_complete.json')
            assert sel['epoch']==expected[1] and sel['lambda_value']==expected[2]
            assert abs(sel['validation_rmse']-expected[0])<1e-12
            assert abs(np.sqrt(np.square(np.load(out/'validation.npy')-val.y).mean())-expected[0])<1e-12
            for file,key in [('selection.json','selection_sha256'),('validation.npy','validation_sha256'),('selected.pt','selected_sha256')]:assert digest(out/file)==done[key]
            assert done['inner_train_rows']==len(inner)
            selection_checks.append(str(out.relative_to(root)))
            return expected[0]
        means={}
        for count in cfg['cluster_counts']:
            values=[]
            for seed in cfg['seeds']:
                out=root/f'c_{count}/cooccurrence/beta_{beta}/seed_{seed}/residual_real'
                assert digest(out/'select_complete.json')==choice['frozen_validation_models'][str(out.relative_to(root))]
                values.append(check_selection(out))
                for phase,group in [('select','inner_train'),('refit','train')]:
                    directory=root/f'c_{count}/clusters/beta_{beta}/seed_{seed}/{phase}'
                    a=np.load(directory/'groups.npz');record=read(directory/'complete.json')
                    old=prior/f'clusters/beta_{beta}/seed_{seed}/{phase}';p=np.load(old/'groups.npz')
                    assert record['groups_sha256']==digest(directory/'groups.npz')
                    assert record['parent_groups_sha256']==digest(old/'groups.npz')
                    fit=np.load(cell/f'data/{group}_users.npy')
                    np.testing.assert_array_equal(a['fit_users'],fit)
                    assert not set(fit)&set(np.load(cell/'data/test_users.npy'))
                    if phase=='select':assert not set(fit)&set(np.load(cell/'data/validation_users.npy'))
                    np.testing.assert_array_equal(a['latent'],p['latent'])
                    counts=np.bincount(api[seed][fit].reshape(-1),minlength=len(a['item_cluster']))
                    np.testing.assert_array_equal(counts,a['incidence_counts'])
                    item_cluster=a['item_cluster'];np.testing.assert_array_equal(item_cluster<0,counts==0)
                    assert np.unique(item_cluster[counts>0]).tolist()==list(range(count))
                    np.testing.assert_array_equal(a['cooccurrence'],item_cluster[api[seed]])
                    np.testing.assert_array_equal(a['cooccurrence']<0,a['size_matched_random']<0)
                    for j in range(count):np.testing.assert_array_equal((a['cooccurrence']==j).sum(1),(a['size_matched_random']==j).sum(1))
                    distance=np.square(a['latent'][:,None,:]-a['centers'][None,:,:]).sum(-1)
                    np.testing.assert_array_equal(distance.argmin(1),item_cluster[a['active_items']])
                    cluster_checks.append(str(directory.relative_to(root)))
            means[count]=float(np.mean(values))
        assert choice['cluster_count']==min(cfg['cluster_counts'],key=lambda c:(means[c],c))
        for row in choice['grid']:assert row['mean_validation_rmse']==means[row['cluster_count']]
        baseline=test.i.map(train.groupby('i').y.mean()).fillna(train.y.mean()).to_numpy()
        summary=read(root/'summary.json')['results'][str(beta)]
        assert abs(np.sqrt(np.square(baseline-test.y).mean())-summary['models']['item_mean']['aggregate']['rmse']['mean'])<1e-12
        for rel,h in frozen['final_models'].items():
            if f'/beta_{beta}/' not in rel:continue
            out=root/rel;assert digest(out/'complete.json')==h
            assert (out/'complete.json').stat().st_mtime_ns<evaluation_time
            complete=read(out/'complete.json');selected=read(out/'select_complete.json');sel=read(out/'selection.json');check_selection(out)
            count=int(rel.split('/')[0][2:]);variant=rel.split('/')[1];arm=out.name.removeprefix('residual_');seed=int(out.parent.name.removeprefix('seed_'))
            assert complete['initial_parameters_sha256']==selected['initial_parameters_sha256']
            assert complete['refit_epochs']==sel['epoch']==len(read(out/'refit_history.json'))
            assert complete['trainable_parameters']-complete['parent_trainable_parameters']==(count-1)*2048
            assert complete['full_outer_train_rows']==len(train)
            for file,key in [('final.pt','final_sha256'),('test.npy','test_sha256'),('selection.json','selection_sha256'),('assignment.npy','assignment_sha256')]:
                assert digest(out/file)==complete[key]
                assert (out/file).stat().st_mtime_ns<evaluation_time
            prediction=np.load(out/'test.npy');assert prediction.shape==(len(test),) and np.isfinite(prediction).all()
            unseen=~test.i.isin(set(train.i)).to_numpy();np.testing.assert_array_equal(prediction[unseen],baseline[unseen])
            expected=summary['models'][f'C{count}_{variant}_{arm}']['per_seed'][cfg['seeds'].index(seed)]
            assert abs(float(np.abs(prediction-test.y).mean())-expected['mae'])<1e-12
            assert abs(float(np.sqrt(np.square(prediction-test.y).mean()))-expected['rmse'])<1e-12
            metric_checks.append(rel)
    result=dict(status='PASS',cluster_fits=len(cluster_checks),selection_records=len(selection_checks),final_models=len(metric_checks),
        candidate_models=36,train_only_clustering=True,validation_only_C_choice=True,independent_metrics_match=True,
        parameter_formula_verified=True,verifier_sha256=digest(Path(__file__)))
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    audit(p.parse_args().root.resolve())
