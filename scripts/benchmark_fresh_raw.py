"""Independent seed-2021 raw-to-result timing; never reuse derived artifacts."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
BENCH = BASE / 'runs/fresh_movie_music_seed2021'
SEED = 2021


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def prepare(root):
    import pandas as pd
    sys.path.insert(0, str(BASE / 'external/PFCDR'))
    from preprocessing import DataPreprocessingReady
    frames = []
    for domain in ['Movies_and_TV', 'CDs_and_Vinyl']:
        raw = BASE / 'data/raw' / f'reviews_{domain}_5.json.gz'
        dest = root / 'data/mid' / f'{domain}.csv'
        assert not dest.exists()
        count = 0
        with gzip.open(raw, 'rt') as source, dest.open('w') as target:
            writer = csv.writer(target)
            writer.writerow(['uid', 'iid', 'y'])
            for line in source:
                row = json.loads(line)
                writer.writerow([row['reviewerID'], row['asin'], row['overall']])
                count += 1
        print('RAW_PARSED', domain, count, 'gzip CRC checked through EOF', flush=True)
        frames.append(pd.read_csv(dest))
    src, tgt = frames
    co = set(src.uid) & set(tgt.uid)
    pair = dict(src='Movies_and_TV', tgt='CDs_and_Vinyl', co_user_num=len(co),
                uid_src=src.uid.nunique(), iid_src=src.iid.nunique(),
                uid_tgt=tgt.uid.nunique(), iid_tgt=tgt.iid.nunique(), topk=70000,
                batchsize_src=512, batchsize_tgt=512, batchsize_meta=512,
                batchsize_map=128, batchsize_test=1024)
    cfg = dict(task='1', base_model='MF', ratio=[.8, .2], seed=SEED, epoch=10,
               lr=.01, lr_prototype=.005, emb_dim=10, wd=0,
               independent_domain_rng=True, src_tgt_pairs={'1': pair})
    save(root / 'configs/first_round.json', cfg)
    save(root / 'configs/blackbox.json', dict(k=100, api='existing_user_id -> top_k_item_ids_and_raw_prediction_scores'))
    for name, vals in [('source_users', list(co) + list(set(src.uid) - co)),
                       ('target_users', list(co) + list(set(tgt.uid) - co)),
                       ('source_items', list(set(src.iid))), ('target_items', list(set(tgt.iid)))]:
        pd.DataFrame(dict(raw_id=vals, id=range(len(vals)))).to_csv(root / f'data/id_maps/{name}.csv', index=False)
    prep = DataPreprocessingReady(str(root / 'data') + '/', {'1': pair}, '1', [.8, .2])
    src, tgt, n = prep.mapper(src, tgt)
    random.seed(SEED)
    original = random.sample
    try:
        random.sample = lambda population, k: original(list(population), k)
        train_src, train_tgt, test, meta, test_ids = prep.split(src, tgt, n)
    finally:
        random.sample = original
    assert not set(train_tgt.uid) & set(test.uid)
    prep.save(train_src, train_tgt, test, meta, test_ids)
    save(root / 'data/split_manifest.json', dict(seed=SEED, pythonhashseed=str(SEED),
         task='Movie->Music', beta=.2, overlap_users=n, cold_users=len(test_ids),
         source_interactions=len(src), target_train_interactions=len(train_tgt), test_interactions=len(test),
         raw_inputs=[str(BASE / 'data/raw' / f'reviews_{d}_5.json.gz') for d in ['Movies_and_TV', 'CDs_and_Vinyl']]))


def worker(arm):
    root = BENCH / arm
    assert os.environ['PYTHONHASHSEED'] == str(SEED)
    for folder in ['data/mid', 'data/id_maps', 'configs', 'checkpoints', 'queries', 'results', 'logs']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    stages = []
    start = time.monotonic()
    save(BENCH / 'status.json', dict(state='running', arm=arm, stage='parse_raw_and_split'))
    prepare(root)
    stages.append(dict(stage='parse_raw_and_split', seconds=time.monotonic() - start))
    env = dict(os.environ, CDR_RUN_ROOT=str(root), OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0')
    steps = ([('base_train', 'experiment.py', ['train']),
              ('wb_invert', 'experiment.py', ['invert', '--source', 'wb']),
              ('wb_bridge', 'experiment.py', ['bridge', '--source', 'wb'])]
             if arm == 'wb' else
             [('source_train', 'experiment.py', ['train_source']),
              ('query_top100', 'export_api.py', []),
              ('list_train', 'run_list_encoder.py', [])])
    for label, script, args in steps:
        save(BENCH / 'status.json', dict(state='running', arm=arm, stage=label))
        start = time.monotonic()
        print('START', label, flush=True)
        with (root / 'logs' / f'{label}.log').open('x') as log:
            subprocess.run([sys.executable, '-u', str(BASE / 'scripts' / script), *args],
                           cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        stages.append(dict(stage=label, seconds=time.monotonic() - start))
        save(root / 'results/stage_timings.json', stages)
        print('DONE', stages[-1], flush=True)


def supervise():
    # Refuse to continue an old run: this benchmark must start clean.
    BENCH.mkdir(parents=True, exist_ok=False)
    code = ['benchmark_fresh_raw.py', 'experiment.py', 'run_list_encoder.py', 'export_api.py', 'batching.py']
    save(BENCH / 'protocol.json', dict(seed=SEED, beta=.2, epochs=10,
         timing='wall clock from worker launch before raw parsing to final predictions/checkpoint/log saving',
         reuse='raw compressed input files and algorithm code only; no derived data, weights, lists or prototypes',
         os_cache='OS file cache is not forcibly flushed; this is not a cold-disk benchmark',
         source_rng='same seed+1 minibatch stream in both independently trained source services',
         scripts_sha256={p: hashlib.sha256((BASE / 'scripts' / p).read_bytes()).hexdigest() for p in code}))
    totals = {}
    for arm in ['wb', 'list']:
        root = BENCH / arm
        root.mkdir()
        env = dict(os.environ, PYTHONHASHSEED=str(SEED), OMP_NUM_THREADS='4')
        start = time.monotonic()
        with (root / 'pipeline.log').open('x') as log:
            subprocess.run([sys.executable, '-u', __file__, '--arm', arm], env=env,
                           cwd=BASE, stdout=log, stderr=subprocess.STDOUT, check=True)
        totals[arm] = time.monotonic() - start
        save(BENCH / 'wall_times.json', totals)
    # Post-run equality checks are outside each method's measured time.
    import numpy as np
    import torch
    wb = np.load(BENCH / 'wb/results/wb_predictions.npz')
    direct = np.load(BENCH / 'list/results/list_encoder_predictions.npz')
    for key in ['uid', 'iid', 'y']:
        np.testing.assert_array_equal(wb[key], direct[key])
    for filename in ['train_src.csv', 'train_tgt.csv', 'test.csv', 'train_meta.csv', 'test_list.csv']:
        rel = Path('data/ready/_8_2/tgt_CDs_and_Vinyl_src_Movies_and_TV') / filename
        assert hashlib.sha256((BENCH / 'wb' / rel).read_bytes()).digest() == hashlib.sha256((BENCH / 'list' / rel).read_bytes()).digest()
    a = torch.load(BENCH / 'wb/checkpoints/victim_source.pt', weights_only=True, map_location='cpu')
    b = torch.load(BENCH / 'list/checkpoints/victim_source.pt', weights_only=True, map_location='cpu')
    equal = all(torch.equal(a[k], b[k]) for k in a)
    assert equal, 'Independent source retraining did not match; inspect before interpreting timings'
    summary = dict(seed=SEED, wall_seconds=totals, speedup=totals['wb']/totals['list'],
                   independently_retrained_source_weights_identical=equal, identical_splits=True, metrics={})
    for label, data in [('wb', wb), ('list', direct)]:
        err = data['prediction'] - data['y']
        summary['metrics'][label] = dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt(np.square(err).mean())))
    save(BENCH / 'comparison.json', summary)
    save(BENCH / 'status.json', dict(state='complete'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['wb', 'list'])
    args = parser.parse_args()
    try:
        worker(args.arm) if args.arm else supervise()
    except Exception as exc:
        if BENCH.exists():
            save(BENCH / 'failure.json', dict(error=repr(exc), arm=args.arm))
        raise
