"""Independent artifact checks for training-only clustering and frozen LIST evaluation."""
import argparse
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from verify_list_segmented import verify as verify_models,read,digest


def verify(root):
    verify_models(root)
    cfg=read(root/'protocol.json');parent=Path(cfg['parent'])
    prior=parent.parent/'paper_list_dnn_8638'/'music_movie_clean'
    evaluation_time=(root/'evaluation_started.json').stat().st_mtime_ns
    checks=[]
    for beta in cfg['betas']:
        cell=parent/f'beta_{beta}'
        for seed in cfg['seeds']:
            query=prior/f'seed_{seed}/queries/DNN/topk_ids.npz'
            ids=np.load(query)['item_ids'];vocab=np.unique(ids);compact=np.searchsorted(vocab,ids)+1
            for phase,group in [('select','inner_train'),('refit','train')]:
                out=root/f'clusters/beta_{beta}/seed_{seed}/{phase}';record=read(out/'complete.json');a=np.load(out/'groups.npz')
                assert record['groups_sha256']==digest(out/'groups.npz')
                assert record['query_sha256']==digest(query)
                assert record['fit_users_sha256']==digest(cell/f'data/{group}_users.npy')
                assert (out/'complete.json').stat().st_mtime_ns<evaluation_time
                users=np.load(cell/f'data/{group}_users.npy');np.testing.assert_array_equal(a['fit_users'],users)
                forbidden=np.load(cell/'data/test_users.npy')
                assert not set(users)&set(forbidden)
                if phase=='select':assert not set(users)&set(np.load(cell/'data/validation_users.npy'))
                counts=np.bincount(compact[users].reshape(-1),minlength=len(vocab)+1)
                np.testing.assert_array_equal(counts,a['incidence_counts'])
                item_cluster=a['item_cluster'];assert ((item_cluster>=-1)&(item_cluster<3)).all()
                np.testing.assert_array_equal(item_cluster<0,counts==0)
                actual=item_cluster[compact];np.testing.assert_array_equal(actual,a['cooccurrence'])
                random=a['size_matched_random'];np.testing.assert_array_equal(actual<0,random<0)
                assert ((random>=-1)&(random<3)).all()
                for j in range(3):np.testing.assert_array_equal((actual==j).sum(1),(random==j).sum(1))
                active=np.flatnonzero(counts);np.testing.assert_array_equal(active,a['active_items'])
                q=(1/np.log2(np.arange(cfg['k'])+2)).astype(np.float32)
                matrix=sparse.csr_matrix((np.tile(q,len(users)),(compact[users].reshape(-1),np.repeat(np.arange(len(users)),cfg['k']))),shape=(len(vocab)+1,len(users)))
                assert a['svd_components'].shape==(cfg['svd_dim'],len(users))
                latent=normalize(normalize(matrix[active])@a['svd_components'].T)
                np.testing.assert_allclose(latent,a['latent'],rtol=2e-3,atol=2e-5)
                distances=np.square(latent[:,None,:]-a['centers'][None,:,:]).sum(-1)
                np.testing.assert_array_equal(distances.argmin(1),item_cluster[active])
                for variant in cfg['variants']:
                    for arm in cfg['arms']:
                        m=root/variant/f'beta_{beta}/seed_{seed}/residual_{arm}'
                        assert read(m/'complete.json')[f'cluster_{phase}_sha256']==record['groups_sha256']
                        old=Path(cfg['segmented_parent'])/'rank_segments'/f'beta_{beta}/seed_{seed}/residual_real/complete.json'
                        assert read(m/'complete.json')['initial_parameters_sha256']==read(old)['initial_parameters_sha256']
                checks.append(dict(beta=beta,seed=seed,phase=phase,fit_users=len(users),items=len(active),status='PASS'))
    result=dict(status='PASS',cluster_fits=len(checks),checks=checks,verifier_sha256=digest(Path(__file__)),
        train_only_cluster_fits=True,full_refit_recomputes_clusters=True,exact_per_user_random_group_sizes=True,
        reconstructed_features_match=True,nearest_centers_match=True,unknown_items_excluded=True,
        initialization_matches_8703=True,model_audit_sha256=digest(root/'independent_verification.json'))
    (root/'cluster_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CLUSTER_INDEPENDENT_VERIFY_PASS',len(checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    verify(p.parse_args().root.resolve())
