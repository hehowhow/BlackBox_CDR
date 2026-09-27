"""Independent CPU audit of frozen response-depth LIST results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from book_movie_protocol import BASE, digest, save, predict
from list_source_backbones import read, check_code


def verify(root):
    cfg = read(root/'protocol.json'); check_code(root)
    parent = Path(cfg['parent']); table = pd.read_csv(root/'per_seed.csv')
    assert read(root/'summary.json')['status'] == 'PASS'
    audits = []; prefix_audits = []
    for seed in cfg['seeds']:
        q = root/f'queries/seed_{seed}'; qr = read(q/'complete.json')
        original = BASE/f'runs/paper_list_dnn_8638/music_movie_clean/seed_{seed}/queries/DNN/topk_ids.npz'
        old = np.load(original)['item_ids']; full = np.load(q/'k_200_ids.npz')['item_ids']
        np.testing.assert_array_equal(full[:, :100], old)
        for k in [20, 50, 100, 200]:
            assert digest(q/f'k_{k}_ids.npz') == qr['ids_sha256'][str(k)]
            np.testing.assert_array_equal(np.load(q/f'k_{k}_ids.npz')['item_ids'], full[:, :k])
        vocab = np.load(q/'vocabulary.npy'); n = qr['old_vocabulary_size']
        np.testing.assert_array_equal(vocab[:n], np.unique(old))
        np.testing.assert_array_equal(np.sort(vocab), np.unique(full))
        prefix_audits.append(dict(seed=seed,status='PASS',prefix_100_exact=True,all_short_prefixes_exact=True))
    for beta in cfg['betas']:
        cell = parent/f'beta_{beta}'
        test = pd.read_csv(cell/'data/test.csv',header=None).to_numpy()
        val = pd.read_csv(cell/'data/validation.csv',header=None).to_numpy()
        train = pd.read_csv(cell/'data/train.csv',header=None,names=['uid','iid','rating'])
        naive_mean = pd.Series(test[:,1].astype(int)).map(train.groupby('iid').rating.mean()).fillna(train.rating.mean()).to_numpy()
        stored_mean = predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean')
        np.testing.assert_allclose(naive_mean,stored_mean,rtol=0,atol=1e-12)
        baseval = predict(dict(np.load(cell/'inner_baseline/statistics.npz')),val[:,1].astype(int),'item_mean')
        initial_rmse = float(np.sqrt(np.square(baseval-val[:,2]).mean()))
        for k in cfg['depths']:
            for seed in cfg['seeds']:
                p = root/f'k_{k}/beta_{beta}/seed_{seed}/residual_real'
                c = read(p/'complete.json'); s = read(p/'selection.json'); history = read(p/'select_history.json')
                chosen = dict(epoch=0,lambda_value=0.,validation_rmse=initial_rmse)
                for h in history:
                    for lam in cfg['lambdas']:
                        value = h['validation'][str(lam)]
                        if value < chosen['validation_rmse']:
                            chosen=dict(epoch=h['epoch'],lambda_value=lam,validation_rmse=value)
                assert s == chosen, (k,beta,seed,s,chosen)
                assert len(history)==10 and len(read(p/'refit_history.json'))==s['epoch']
                pred=np.load(p/'test.npy'); y=test[:,2]
                mae=float(np.abs(pred-y).mean());rmse=float(np.sqrt(np.square(pred-y).mean()))
                row=table[(table.variant==f'k_{k}')&(table.beta==beta)&(table.seed==seed)].iloc[0]
                np.testing.assert_allclose([mae,rmse],[row.mae,row.rmse],rtol=0,atol=1e-12)
                reference=np.load(cell/f'seed_{seed}/residual_real/test.npy')
                delta=[mae-float(np.abs(reference-y).mean()),rmse-float(np.sqrt(np.square(reference-y).mean()))]
                np.testing.assert_allclose(delta,[row.delta_mae_vs_original,row.delta_rmse_vs_original],rtol=0,atol=1e-12)
                assert c['test_labels_opened'] is False and c['own_k_list_only'] is True
                for filename,key in [('final.pt','final_sha256'),('selection.json','selection_sha256'),('test.npy','test_sha256')]:
                    assert digest(p/filename)==c[key]
                assert (p/'final.pt').stat().st_mtime_ns < (root/'evaluation_started.json').stat().st_mtime_ns
                audits.append(dict(variant=f'k_{k}',beta=beta,seed=seed,mae=mae,rmse=rmse,status='PASS'))
    result=dict(status='PASS',audited_models=len(audits),model_audits=audits,prefix_audits=prefix_audits,
        independent_metrics_and_deltas=True,independent_item_mean=True,validation_selection_replayed=True,
        all_models_frozen_before_evaluation=True,model_and_prediction_hashes=True)
    save(root/'independent_verification.json',result)
    print('ACQUISITION_INDEPENDENT_VERIFICATION_PASS',len(audits),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    verify(p.parse_args().root)
