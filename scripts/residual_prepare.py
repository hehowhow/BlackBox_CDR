"""Prepare all three fixed cold-user tasks before any model/test evaluation."""
import argparse
import json
from pathlib import Path
import random
import shutil
import numpy as np
import pandas as pd
from book_movie_protocol import BASE, fit, predict, metric, digest, save

TASKS = {'book_movie': ('Books', 'Movies_and_TV'),
         'movie_book': ('Movies_and_TV', 'Books'),
         'movie_music_clean': ('Movies_and_TV', 'CDs_and_Vinyl')}


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    config = dict(seeds=[2020, 2021, 2022], source_epochs=10, source_dim=10,
                  source_lr=.01, source_batch=512, k=100, student_epochs=20,
                  student_dim=32, student_hidden=64, student_lr=.001,
                  student_weight_decay=.00001, student_dropout=.1, student_batch=2048,
                  arms=['target_bias', 'real', 'shuffled'], selection_metric='validation_rmse',
                  include_epoch_zero=True, baseline_alphas=[0, 1, 5, 10, 20, 50, 100],
                  objective='target training MSE', permutation='fixed derangement within each train/validation/test user group',
                  bootstrap_repeats=2000, bootstrap_seed=991,
                  test_policy='After dataset construction, only aggregate stage reads test labels, after all nine runs have frozen selections',
                  source_api='Fresh MF Top100 opaque IDs only; no scores/weights/histories passed to student',
                  split_note='One fixed user split per task; three training/permutation seeds, not three independent splits',
                  code_sha256={name: digest(BASE/'scripts'/name) for name in
                               ['residual_prepare.py', 'residual_experiment.py', 'residual_summary.py']})
    save(root/'experiment.json', config)
    for name, (src_name, tgt_name) in TASKS.items():
        task = root/name
        data = task/'data'
        data.mkdir(parents=True)
        for folder in ['runs', 'baseline']:
            (task/folder).mkdir()
        if name == 'book_movie':
            prior = BASE/'runs/book_movie_protocol_8564'
            p = json.loads((prior/'protocol.json').read_text())
            for file, expected in p['data_sha256'].items():
                assert digest(prior/file) == expected
            src = pd.read_csv(prior/'data/train_src.csv', header=None, names=['uid', 'iid', 'y'])
            train = pd.read_csv(prior/'data/train_tgt.csv', header=None, names=['uid', 'iid', 'y'])
            val = pd.read_csv(prior/'data/validation.csv', header=None, names=['uid', 'iid', 'y'])
            test = pd.read_csv(prior/'data/test.csv', header=None, names=['uid', 'iid', 'y'])
            groups = {g: pd.read_csv(prior/f'data/{g}_users.csv', header=None).to_numpy().ravel()
                      for g in ['train', 'validation', 'test']}
            maps = {m: pd.read_csv(prior/f'data/id_maps/{m}.csv', dtype={'raw_id': str})
                    for m in ['source_users', 'target_users', 'source_items', 'target_items']}
            cleanup = dict(rule='No filtering: source and target raw item catalogs already disjoint', removed_items=0, removed_source_ratings=0,
                           preserved_prior_protocol=True, prior_test_sha256=digest(prior/'data/test.csv'))
            overlap = p['overlap_users']
        else:
            src = pd.read_csv(BASE/f'data/mid/{src_name}.csv')
            tgt = pd.read_csv(BASE/f'data/mid/{tgt_name}.csv')
            common = set(src.iid) & set(tgt.iid)
            before_users, before_rows = src.uid.nunique(), len(src)
            overlap_before = len(set(src.uid) & set(tgt.uid))
            if name == 'movie_music_clean':
                src = src[~src.iid.isin(common)].copy()
            else:
                assert not common
            remaining = set(src.iid) & set(tgt.iid)
            assert not remaining
            cleanup = dict(rule='Drop all shared raw item IDs from source before user splitting; target untouched; no reapplication of 5-core',
                           removed_items=len(common), removed_source_ratings=before_rows-len(src),
                           source_users_before=int(before_users), source_users_after=int(src.uid.nunique()),
                           overlap_users_before=overlap_before, remaining_cross_domain_raw_items=len(remaining))
            save(task/'removed_source_items.json', sorted(common))
            co = sorted(set(src.uid) & set(tgt.uid))
            overlap = len(co)
            maps = {}
            for label, values in [('source_users', co+sorted(set(src.uid)-set(co))),
                                  ('target_users', co+sorted(set(tgt.uid)-set(co))),
                                  ('source_items', sorted(set(src.iid))), ('target_items', sorted(set(tgt.iid)))]:
                maps[label] = pd.DataFrame({'raw_id': values, 'id': range(len(values))})
            for frame, prefix in [(src, 'source'), (tgt, 'target')]:
                for col, suffix in [('uid', 'users'), ('iid', 'items')]:
                    m = maps[f'{prefix}_{suffix}']
                    frame[col] = frame[col].map(dict(zip(m.raw_id, m.id)))
            test_ids = sorted(random.Random(2020).sample(list(range(overlap)), round(.2*overlap)))
            rest = sorted(set(range(overlap))-set(test_ids))
            val_ids = sorted(random.Random(20220922).sample(rest, round(.2*overlap)))
            groups = dict(train=np.array(sorted(set(rest)-set(val_ids))), validation=np.array(val_ids), test=np.array(test_ids))
            train = tgt[~tgt.uid.isin(val_ids+test_ids)].copy()
            val, test = tgt[tgt.uid.isin(val_ids)].copy(), tgt[tgt.uid.isin(test_ids)].copy()
        assert not set(maps['source_items'].raw_id) & set(maps['target_items'].raw_id)
        assert not set(train.uid) & (set(val.uid) | set(test.uid))
        assert not set(val.uid) & set(test.uid)
        assert set(train.uid) & set(range(overlap)) == set(groups['train'])
        assert set(val.uid) == set(groups['validation']) and set(test.uid) == set(groups['test'])
        assert set(groups['train']) | set(groups['validation']) | set(groups['test']) == set(range(overlap))
        for frame in [src, train, val, test]:
            assert not frame.duplicated(['uid', 'iid']).any()
            assert frame.y.between(1, 5).all()
        (data/'id_maps').mkdir()
        for label, frame in maps.items():
            assert frame.raw_id.is_unique and frame.id.is_unique
            frame.to_csv(data/f'id_maps/{label}.csv', index=False)
        np.testing.assert_array_equal(maps['source_users'].raw_id.iloc[:overlap], maps['target_users'].raw_id.iloc[:overlap])
        for label, frame in [('source', src), ('train', train), ('validation', val), ('test', test)]:
            frame.to_csv(data/f'{label}.csv', index=False, header=False)
        for label, users in groups.items():
            np.save(data/f'{label}_users.npy', users.astype(np.int64))
        np.save(data/'test_features.npy', test[['uid', 'iid']].to_numpy(dtype=np.int64))
        stats = fit(train)
        np.savez(task/'baseline/statistics.npz', **stats)
        candidates = ['item_mean']+[f'smoothed_mean_{a}' for a in config['baseline_alphas'] if a]
        scores = {k: metric(predict(stats, val.iid.to_numpy(), k), val) for k in candidates+['item_median', 'global_mean']}
        selected = min(candidates, key=lambda k: scores[k]['rmse'])
        catalog_baseline = predict(stats, np.arange(len(maps['target_items'])), selected)
        np.save(task/'baseline/values.npy', catalog_baseline.astype(np.float32))
        seen = np.isin(np.arange(len(maps['target_items'])), stats['ids'])
        np.save(task/'baseline/seen.npy', seen)
        save(task/'baseline/selection.json', dict(selected=selected, validation_metrics=scores, selection='validation RMSE; means only',
                                                 training_only_statistics=True, statistics_sha256=digest(task/'baseline/statistics.npz')))
        manifest = dict(task=name, source=src_name, target=tgt_name, overlap_users=overlap,
                        source_users=len(maps['source_users']), source_items=len(maps['source_items']),
                        target_users=len(maps['target_users']), target_items=len(maps['target_items']),
                        rows=dict(source=len(src), train=len(train), validation=len(val), test=len(test)),
                        users={k: len(v) for k, v in groups.items()}, cleanup=cleanup,
                        cross_domain_raw_item_overlap=0, cross_domain_same_user_item_rating_leaks=0,
                        test_unseen_item_rows=int((~np.isin(test.iid, stats['ids'])).sum()),
                        split_policy='60/20/20 overlap users; all target-only users in training; no temporal filtering',
                        hashes={str(p.relative_to(task)): digest(p) for p in sorted(data.rglob('*')) if p.is_file()},
                        baseline_hashes={str(p.relative_to(task)): digest(p) for p in sorted((task/'baseline').glob('*')) if p.is_file()})
        save(task/'manifest.json', manifest)
        print('DATA_PREPARED', name, json.dumps({k: v for k, v in manifest.items() if 'hash' not in k}), flush=True)
    print('ALL_DATA_PREPARED', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    prepare(parser.parse_args().root.resolve())
