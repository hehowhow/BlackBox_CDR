"""Independent model-selection and native-forward checks before test evaluation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pfcdr_backbone_study import BACKBONES, read, make_model
from book_movie_protocol import digest, save
from residual_experiment import guard_test_labels


def verify(root):
    guard_test_labels(root)
    cfg, manifest = read(root/'protocol.json'), read(root/'manifest.json')
    assert read(root/'selftest.json')['status'] == 'PASS'
    for p,h in {**cfg['code_sha256'],**cfg['upstream_sha256']}.items(): assert digest(Path(p)) == h
    val = pd.read_csv(root/'data/validation.csv', header=None).to_numpy()
    train = np.load(root/'data/train_users.npy')
    groups = [set(np.load(root/f'data/{g}_users.npy')) for g in ['train','validation','test']]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    base, seen = np.load(root/'baseline/values.npy'), np.load(root/'baseline/seen.npy')
    rows = np.random.default_rng(752).choice(len(val), size=32, replace=False)
    checks=[]
    initialization=None
    for name in BACKBONES:
        out=root/name
        initial=read(out/'initialization.json')
        if initialization is None: initialization=initial
        assert initial == initialization
        done=read(out/'complete.json')
        assert done['status']=='PASS'
        assert digest(out/'pretrained.pt')==done['pretrained_sha256']
        assert digest(out/'selection.json')==done['selection_sha256']
        for d,h in done['prototypes'].items(): assert digest(out/f'prototypes_{d}.npz')==h
        src,tgt=np.load(out/'prototypes_src.npz'),np.load(out/'prototypes_tgt.npz')
        np.testing.assert_array_equal(tgt['user_ids'],train)
        np.testing.assert_array_equal(src['user_ids'],np.arange(manifest['overlap_users']))
        assert np.isfinite(src['prototypes']).all() and np.isfinite(tgt['prototypes']).all()
        ph=read(out/'pretrain_history.json')
        for d in ['src','tgt']:
            assert [r['epoch'] for r in ph if r['domain']==d]==list(range(1,cfg['pretrain_epochs']+1))
            ih=read(out/f'inversion_{d}_history.json')
            assert all(r['iterations']==cfg['inversion_iterations'] for r in ih)
            assert sum(r['users'] for r in ih)==(len(src['user_ids']) if d=='src' else len(train))
        history=read(out/'bridge_history.json')
        assert [r['epoch'] for r in history]==list(range(1,cfg['bridge_epochs']+1))
        model=make_model(name,manifest,cfg['dim'])
        model.load_state_dict(torch.load(out/'pretrained.pt',map_location='cuda',weights_only=True))
        model.eval()
        for k,sel in read(out/'selection.json').items():
            assert min(history,key=lambda r:r['validation'][k]['rmse'])['epoch']==sel['epoch']
            assert digest(out/f'bridge_k{k}.pt')==sel['checkpoint_sha256']
            assert digest(out/f'validation_k{k}.npy')==sel['validation_sha256']
            assert digest(out/f'test_k{k}.npy')==sel['test_sha256']
            assert digest(out/f'test_raw_k{k}.npy')==sel['raw_test_sha256']
            pv=np.load(out/f'validation_k{k}.npy')
            assert abs(float(np.sqrt(np.square(pv-val[:,2]).mean()))-sel['validation_rmse'])<1e-12
            assert np.isfinite(np.load(out/f'test_k{k}.npy')).all()
            # Direct published forward is an independent oracle for its native K.
            native_k=50000 if name=='GMF' else 70000
            if int(k)==native_k:
                model.rp_mapping.load_state_dict(torch.load(out/f'bridge_k{k}.pt',map_location='cuda',weights_only=True))
                proto=src['prototypes'][val[rows,0].astype(int)]
                x=torch.tensor(np.column_stack([val[rows,1],proto]),dtype=torch.float32,device='cuda')
                with torch.no_grad(): pred=torch.cat([model(b,'test_source_free') for b in x.split(8)]).cpu().numpy()
                items=val[rows,1].astype(int)
                pred=np.where(seen[items],pred,base[items])
                np.testing.assert_allclose(pred,pv[rows],rtol=5e-5,atol=5e-5)
                checks.append(dict(backbone=name,k=int(k),native_prediction_samples=len(rows),passed=True,
                                   maximum_native_difference=float(np.abs(pred-pv[rows]).max())))
            checks.append(dict(backbone=name,k=int(k),epoch=sel['epoch'],selection_and_hashes=True))
        del model
        torch.cuda.empty_cache()
    save(root/'verification.json',dict(status='PASS',checks=checks,
         scope='Native model prediction samples, full stored validation metrics/selection, all model/data provenance; not repeated training',
         shared_embedding_initialization_verified=True,target_prototypes_only_training_users=True))
    print('BACKBONE_VERIFICATION_PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    verify(p.parse_args().root.resolve())
