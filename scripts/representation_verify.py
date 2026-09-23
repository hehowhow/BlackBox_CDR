"""Independently reconstruct diagnostic features and their train-only scaling."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from book_movie_protocol import BASE, digest, save


def main(root):
    cfg = json.loads((root/'protocol.json').read_text())
    prior = Path(cfg['prior_root'])
    checks = []
    for name in ['book_movie', 'movie_book', 'movie_music_clean']:
        task = root/name
        m = json.loads((task/'manifest.json').read_text())
        records = json.loads((task/'features/manifest.json').read_text())
        users = np.load(task/'data/train_users.npy')
        source = pd.read_csv(task/'data/source.csv', header=None, names=['uid', 'iid', 'y'])
        co = source[source.uid < m['overlap_users']].copy()
        by_user = co.groupby('uid').y
        item_mean = source.groupby('iid').y.mean()
        item_count = source.groupby('iid').size()
        counts = by_user.size()
        co['centered'] = co.y-co.iid.map(item_mean)
        co['popularity'] = np.log1p(co.iid.map(item_count))
        fractions = pd.crosstab(co.uid, co.y).reindex(columns=[1, 2, 3, 4, 5], fill_value=0).div(counts, axis=0)
        expected = np.column_stack([np.log1p(counts), by_user.mean(), by_user.std(ddof=0), fractions.to_numpy(),
                                    co.groupby('uid').centered.mean(), co.groupby('uid').popularity.mean()])
        for label, record in records.items():
            file = task/'features'/f'{label}.npz'
            assert digest(file) == record['sha256']
            a = np.load(file)
            if label == 'history_stats':
                np.testing.assert_allclose(a['raw'], expected, rtol=1e-10, atol=1e-10)
            else:
                seed = int(label.rsplit('_', 1)[1])
                p = prior/name/f'runs/seed_{seed}/source.pt'
                assert digest(p) == record['inputs']['source_checkpoint_sha256']
                state = torch.load(p, map_location='cpu', weights_only=True)
                np.testing.assert_array_equal(a['raw'], state['users.weight'][:m['overlap_users']].numpy())
            center = a['raw'][users].mean(0)
            scale = np.maximum(a['raw'][users].std(0), 1e-6)
            np.testing.assert_array_equal(a['center'], center)
            np.testing.assert_array_equal(a['scale'], scale)
            normalized = np.pad((a['raw']-center)/scale, ((0, 0), (0, 32-len(center)))).astype(np.float32)
            np.testing.assert_array_equal(a['features'], normalized)
            checks.append(dict(task=name, features=label, raw_features_verified=True, training_only_scaling_verified=True))
    save(root/'feature_verification.json', dict(status='PASS', checks=checks))
    print('INDEPENDENT_FEATURE_VERIFICATION_PASS', len(checks), 'feature snapshots')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
