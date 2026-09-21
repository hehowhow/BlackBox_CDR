"""Read-only experiment audit; diagnostics are written to a separate audit folder."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'results/list_audit'
OUT.mkdir(exist_ok=True)


def metric(pred, y):
    err = np.asarray(pred) - y
    return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt(np.square(err).mean())))


def rows_hash(data):
    return np.sort(pd.util.hash_pandas_object(pd.DataFrame(data, columns=['uid', 'iid', 'y']).astype('int64'), index=False).to_numpy())


def audit(root):
    tick = time.monotonic()
    cfg = json.loads((root / 'configs/first_round.json').read_text())
    pair = cfg.get('src_tgt_pairs', {}).get('1', dict(src='Books', tgt='Movies_and_TV'))
    data = root / f"data/ready/_8_2/tgt_{pair['tgt']}_src_{pair['src']}"
    train = pd.read_csv(data / 'train_tgt.csv', header=None).to_numpy(dtype=np.int64)
    test = pd.read_csv(data / 'test.csv', header=None).to_numpy(dtype=np.int64)
    meta = pd.read_csv(data / 'train_meta.csv', header=None).to_numpy(dtype=np.int64)
    api = np.load(root / 'queries/topk_scores.npz')
    manifest = json.loads((root / 'data/split_manifest.json').read_text())
    n = manifest['overlap_users']
    assert not np.intersect1d(train[:, 0], test[:, 0]).size
    assert not np.intersect1d(meta[:, 0], test[:, 0]).size
    np.testing.assert_array_equal(rows_hash(meta), rows_hash(train[train[:, 0] < n]))
    cold = pd.read_csv(data / 'test_list.csv', header=None).to_numpy().ravel()
    np.testing.assert_array_equal(np.sort(cold), np.unique(test[:, 0]))
    user_map = pd.read_csv(root / 'data/id_maps/target_users.csv', dtype={'raw_id': str})
    src_users = pd.read_csv(root / 'data/id_maps/source_users.csv', dtype={'raw_id': str})
    for frame in [user_map, src_users]:
        assert frame.raw_id.is_unique and frame.id.is_unique
    np.testing.assert_array_equal(user_map.raw_id.to_numpy()[:n], src_users.raw_id.to_numpy()[:n])
    target_items = pd.read_csv(root / 'data/id_maps/target_items.csv', dtype={'raw_id': str})
    source_items = pd.read_csv(root / 'data/id_maps/source_items.csv', dtype={'raw_id': str})
    raw_overlap = set(source_items.raw_id) & set(target_items.raw_id)
    # Reconstruct target ratings independently from the pre-split triplets.
    mid = root / f"data/mid/{pair['tgt']}.csv"
    if not mid.exists():
        mid = BASE / f"data/mid/{pair['tgt']}.csv"
    original = pd.read_csv(mid, dtype={'uid': str, 'iid': str})
    original.uid = original.uid.map(dict(zip(user_map.raw_id, user_map.id)))
    original.iid = original.iid.map(dict(zip(target_items.raw_id, target_items.id)))
    assert not original.isna().any().any()
    original = original.to_numpy(dtype=np.int64)
    is_test = np.isin(original[:, 0], cold)
    np.testing.assert_array_equal(rows_hash(original[is_test]), rows_hash(test))
    np.testing.assert_array_equal(rows_hash(original[~is_test]), rows_hash(train))
    del original, is_test
    artifact = np.load(root / 'results/list_encoder_predictions.npz')
    for key, col in [('uid', 0), ('iid', 1), ('y', 2)]:
        np.testing.assert_array_equal(artifact[key], test[:, col])
    assert artifact['prediction'].shape == artifact['y'].shape == (len(test),)
    assert np.isfinite(artifact['prediction']).all()
    metrics = dict(list=metric(artifact['prediction'], test[:, 2]))
    mean = float(train[:, 2].mean())
    median = float(np.median(train[:, 2]))
    metrics['target_train_global_mean'] = metric(mean, test[:, 2])
    metrics['target_train_global_median'] = metric(median, test[:, 2])
    count = np.bincount(train[:, 1])
    sums = np.bincount(train[:, 1], weights=train[:, 2], minlength=len(count))
    means = np.divide(sums, count, out=np.full(len(count), mean), where=count != 0)
    metrics['target_train_item_mean'] = metric(means[test[:, 1]], test[:, 2])
    state = torch.load(root / 'checkpoints/list_encoder.pt', weights_only=True, map_location='cpu')
    sd = state['state_dict']
    assert abs(float(sd['mean']) - mean) < 1e-6
    metrics['list_learned_mean_plus_item_bias'] = metric(float(sd['mean']) + sd['item_bias.weight'].numpy()[test[:, 1], 0], test[:, 2])
    vocab = np.asarray(state['source_vocabulary'])
    compact = np.searchsorted(vocab, api['item_ids']) + 1
    np.testing.assert_array_equal(vocab[compact - 1], api['item_ids'])
    np.testing.assert_array_equal(api['user_ids'], np.arange(n))
    if 'lists' in sd:
        np.testing.assert_array_equal(sd['lists'].numpy()[:n], compact)
    # Independent forward: explicit gather/weighted sum, not production EmbeddingBag.
    weights = sd['rank_weights'].cuda()
    tensors = {k: v.cuda() for k, v in sd.items() if k != 'lists'}
    lists = torch.tensor(compact, device='cuda', dtype=torch.long)
    def forward(u, i):
        h = (tensors['source_items.weight'][lists[u]] * weights[None, :, None]).sum(1)
        z = F.linear(F.relu(F.linear(h, tensors['encoder.0.weight'], tensors['encoder.0.bias'])),
                     tensors['encoder.3.weight'], tensors['encoder.3.bias'])
        return tensors['mean'] + tensors['item_bias.weight'][i, 0] + (z * tensors['target_items.weight'][i]).sum(1)
    with torch.no_grad():
        predictions = torch.cat([forward(torch.tensor(b[:, 0], device='cuda'), torch.tensor(b[:, 1], device='cuda'))
                                 for b in np.array_split(test, max(1, (len(test)+2047)//2048))]).cpu().numpy()
    np.testing.assert_allclose(predictions, artifact['prediction'], atol=1e-5, rtol=1e-5)
    # Inference-only sensitivity check, not a retrained permutation ablation.
    rng = np.random.default_rng(881)
    mapping = np.arange(n)
    mapping[cold] = rng.permutation(cold)
    with torch.no_grad():
        shuffled = torch.cat([forward(torch.tensor(mapping[b[:, 0]], device='cuda'), torch.tensor(b[:, 1], device='cuda'))
                              for b in np.array_split(test, max(1, (len(test)+2047)//2048))]).cpu().numpy()
    metrics['shuffle_cold_user_lists_inference_only'] = metric(shuffled, test[:, 2])
    # Source-side privileged audit of cached API outputs against actual source MF.
    teacher = torch.load(root / 'checkpoints/victim_source.pt', weights_only=True, map_location='cuda')
    selected = rng.choice(n, size=32, replace=False)
    with torch.no_grad():
        score = teacher['uid_embedding.weight'][torch.tensor(selected, device='cuda')] @ teacher['iid_embedding.weight'].T
        returned = torch.tensor(api['item_ids'][selected].astype(np.int64), device='cuda')
        actual_scores = score.gather(1, returned).cpu().numpy()
        topvalues = score.topk(100, dim=1).values.cpu().numpy()
    np.testing.assert_allclose(actual_scores, api['scores'][selected], atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(actual_scores, topvalues, atol=1e-4, rtol=1e-5)
    result = dict(root=str(root), task=f"{pair['src']}->{pair['tgt']}", seed=cfg['seed'],
                  cold_users=len(cold), test_ratings=len(test), train_ratings=len(train),
                  target_train_test_user_overlap=0, target_pre_split_triplets_exact=True,
                  overlapping_raw_item_ids=len(raw_overlap), overlap_examples=sorted(raw_overlap)[:10],
                  source_and_target_overlap_user_ids_aligned=True, api_sample_verified_users=32,
                  prediction_reconstruction_max_abs_diff=float(np.max(np.abs(predictions-artifact['prediction']))),
                  target_train_mean=mean, target_train_median=median,
                  target_test_fraction_rating5=float(np.mean(test[:, 2] == 5)), metrics=metrics,
                  seconds=time.monotonic()-tick)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for root in [BASE, BASE/'runs/movie_to_book', BASE/'runs/movie_to_music', BASE/'runs/fresh_movie_music_seed2021/list']:
        results.append(audit(root))
        (OUT / 'checks.json').write_text(json.dumps(results, indent=2))
