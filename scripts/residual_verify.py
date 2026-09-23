"""Independent post-run checks of validation selection and data/model snapshots."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from book_movie_protocol import digest, save


def main(root):
    cfg = json.loads((root/'experiment.json').read_text())
    records = []
    for name in ['book_movie', 'movie_book', 'movie_music_clean']:
        task = root/name
        manifest = json.loads((task/'manifest.json').read_text())
        for relative, expected in manifest['hashes'].items():
            assert digest(task/relative) == expected, relative
        for relative, expected in manifest['baseline_hashes'].items():
            assert digest(task/relative) == expected, relative
        src = pd.read_csv(task/'data/id_maps/source_items.csv', dtype={'raw_id': str})
        tgt = pd.read_csv(task/'data/id_maps/target_items.csv', dtype={'raw_id': str})
        assert not set(src.raw_id) & set(tgt.raw_id)
        groups = {g: np.load(task/f'data/{g}_users.npy') for g in ['train', 'validation', 'test']}
        assert not set(groups['train']) & set(groups['validation'])
        assert not set(groups['train']) & set(groups['test'])
        assert not set(groups['validation']) & set(groups['test'])
        val = pd.read_csv(task/'data/validation.csv', header=None).to_numpy()
        y = val[:, 2]
        for seed in cfg['seeds']:
            out = task/f'runs/seed_{seed}'
            selection = json.loads((out/'selection.json').read_text())
            assert selection['real']['initial_state_sha256'] == selection['shuffled']['initial_state_sha256']
            perm = np.load(out/'permutation.npy')
            for users in groups.values():
                np.testing.assert_array_equal(np.sort(perm[users]), np.sort(users))
                assert np.all(perm[users] != users)
            for arm in cfg['arms']:
                history = json.loads((out/f'{arm}_training.json').read_text())
                assert [r['epoch'] for r in history] == list(range(cfg['student_epochs']+1))
                best = min(history, key=lambda r: r['validation_rmse'])
                assert best['epoch'] == selection[arm]['epoch']
                predictions = np.load(out/f'{arm}_validation_predictions.npy')
                actual = float(np.sqrt(np.square(predictions-y).mean()))
                assert abs(actual-selection[arm]['validation_rmse']) < 1e-12
                assert digest(out/f'{arm}.pt') == selection[arm]['checkpoint_sha256']
                assert digest(out/f'{arm}_test_predictions.npy') == selection[arm]['predictions_sha256']
                records.append(dict(task=name, seed=seed, arm=arm, selected_epoch=best['epoch'],
                                    restored_checkpoint_validation_rmse=actual, passed=True))
    result = dict(status='PASS', checks=['all data and baseline hashes unchanged',
                                       'source and target raw item catalogs disjoint',
                                       'user groups disjoint', 'within-split derangements verified',
                                       'paired model initialization hashes identical',
                                       'all arms trained full declared budget',
                                       'selected epochs independently match validation minima',
                                       'restored-checkpoint validation predictions match selected RMSE',
                                       'selected model and test-prediction hashes unchanged'], models=records)
    save(root/'verification.json', result)
    print('INDEPENDENT_VERIFICATION_PASS', len(records), 'models')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
