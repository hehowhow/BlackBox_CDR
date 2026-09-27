"""Read-only independent audit of pair-memory artifacts and metric tables."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from book_movie_protocol import BASE, digest, save, predict


def read(path):
    return json.loads(path.read_text())


def verify(root):
    cfg=read(root/'protocol.json')
    parent=Path(cfg['parent'])
    frozen=read(root/'evaluation_started.json')
    evaluation_ns=(root/'evaluation_started.json').stat().st_mtime_ns
    per_seed=list(csv.DictReader((root/'per_seed.csv').open()))
    comparisons=list(csv.DictReader((root/'comparison.csv').open()))
    verified=[]
    for beta in cfg['betas']:
        cell=parent/f'beta_{beta}'
        test=pd.read_csv(cell/'data/test.csv',header=None).to_numpy()
        np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
        for variant,depth in cfg['variants'].items():
            values=[]
            for seed in cfg['seeds']:
                out=root/variant/f'beta_{beta}/seed_{seed}/residual_real'
                complete=read(out/'complete.json')
                assert digest(out/'complete.json')==frozen[str(out.relative_to(root))]
                assert (out/'complete.json').stat().st_mtime_ns < evaluation_ns
                for file,key in [('test.npy','test_sha256'),('final.pt','final_sha256'),('selection.json','selection_sha256')]:
                    assert digest(out/file)==complete[key]
                original=read(cell/f'seed_{seed}/residual_real/complete.json')
                assert complete['shared_initial_parameters_sha256']==original['initial_parameters_sha256']
                peer=read(root/next(iter(cfg['variants']))/f'beta_{beta}/seed_{seed}/residual_real/complete.json')
                assert complete['initial_parameters_sha256']==peer['initial_parameters_sha256']
                assert complete['trainable_parameters']==peer['trainable_parameters']
                selection=read(out/'selection.json')
                val=pd.read_csv(cell/'data/validation.csv',header=None).to_numpy()
                stats=dict(np.load(cell/'inner_baseline/statistics.npz'))
                base=predict(stats,val[:,1].astype(int),'item_mean')
                best=(float(np.sqrt(np.square(base-val[:,2]).mean())),0,0.)
                for epoch in read(out/'select_history.json'):
                    for lam,score in epoch['validation'].items():
                        if score < best[0]:
                            best=(score,epoch['epoch'],float(lam))
                assert (selection['validation_rmse'],selection['epoch'],selection['lambda_value'])==best
                assert len(read(out/'refit_history.json'))==selection['epoch']
                np.testing.assert_allclose(np.sqrt(np.square(np.load(out/'validation.npy')-val[:,2]).mean()),best[0],rtol=0,atol=1e-12)
                memory_info=read(out/'memory_info.json')
                for phase,name in [('select','inner_train'),('refit','train')]:
                    archive=out/f'{phase}_memory.npz'
                    assert digest(archive)==memory_info[phase]['dictionary_sha256']
                    memory=np.load(archive)
                    np.testing.assert_array_equal(memory['training_users'],np.load(cell/f'data/{name}_users.npy'))
                    assert not np.intersect1d(memory['training_users'],np.load(cell/'data/test_users.npy')).size
                    if phase=='select':
                        assert not np.intersect1d(memory['training_users'],np.load(cell/'data/validation_users.npy')).size
                    keys,support,rows=memory['keys'],memory['support'],memory['pair_rows']
                    assert len(keys)<=cfg['memory_capacity']
                    assert np.all(support>=cfg['memory_minimum_users'])
                    assert rows.min()>=0 and rows.max()<=len(keys)
                    # The support count equals the number of training users assigned a key.
                    observed=np.bincount(rows[memory['training_users']].ravel(),minlength=len(keys)+1)[1:]
                    np.testing.assert_array_equal(observed,support)
                    assert rows.shape[1]<=depth*(depth-1)//2
                pred=np.load(out/'test.npy')
                old=np.load(cell/f'seed_{seed}/residual_real/test.npy')
                value=dict(mae=float(np.abs(pred-test[:,2]).mean()),rmse=float(np.sqrt(np.square(pred-test[:,2]).mean())))
                row=next(r for r in per_seed if r['variant']==variant and int(r['beta'])==beta and int(r['seed'])==seed)
                for key,expected in value.items():
                    assert abs(float(row[key])-expected)<1e-12
                assert abs(float(row['delta_rmse'])-(value['rmse']-np.sqrt(np.square(old-test[:,2]).mean())))<1e-12
                values.append(value)
                verified.append(str(out.relative_to(root)))
            row=next(r for r in comparisons if r['variant']==variant and int(r['beta'])==beta)
            for key in ['mae','rmse']:
                assert abs(float(row[key])-np.mean([v[key] for v in values]))<1e-12
    save(root/'independent_verification.json',dict(status='PASS',models=len(verified),
        checks=['hashes and freeze before evaluation','archived shared initialization','validation-only epoch/lambda optimum',
            'identical parameter initialization across memory variants','selected epoch full refit','phase-only dictionary users','dictionary support matches training-user hits','metric recomputation'],
        verifier_sha256=digest(Path(__file__)),verified=verified))
    print('MEMORY_INDEPENDENT_VERIFICATION_PASS',len(verified),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    verify(parser.parse_args().root.resolve())
