"""Isolated, restartable fixed-setting Movie->Book and Movie->Music runs."""
import csv
import gzip
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'runs'
sys.path.insert(0, str(BASE / 'external/PFCDR'))
from preprocessing import DataPreprocessingReady
from download_data import download


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def prepare(name, target):
    root = OUT / name
    for folder in ['data', 'data/id_maps', 'configs', 'logs', 'results', 'queries', 'checkpoints']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    if (root / 'data/split_manifest.json').exists():
        return root
    random.seed(2020)
    src = pd.read_csv(BASE / 'data/mid/Movies_and_TV.csv')
    tgt = pd.read_csv(BASE / f'data/mid/{target}.csv')
    pair = dict(src='Movies_and_TV', tgt=target, co_user_num=len(set(src.uid) & set(tgt.uid)),
                uid_src=src.uid.nunique(), iid_src=src.iid.nunique(),
                uid_tgt=tgt.uid.nunique(), iid_tgt=tgt.iid.nunique(), topk=70000,
                batchsize_src=512, batchsize_tgt=512, batchsize_meta=512,
                batchsize_map=128, batchsize_test=1024)
    cfg = json.loads((BASE / 'configs/first_round.json').read_text())
    cfg.update(task='1', src_tgt_pairs={'1': pair})
    save(root / 'configs/first_round.json', cfg)
    bb = json.loads((BASE / 'configs/blackbox.json').read_text())
    bb['query_users'] = f"all {pair['co_user_num']} overlapping users, including cold target users"
    save(root / 'configs/blackbox.json', bb)
    co = set(src.uid) & set(tgt.uid)
    for label, values in [('source_users', list(co) + list(set(src.uid) - co)),
                          ('target_users', list(co) + list(set(tgt.uid) - co)),
                          ('source_items', list(set(src.iid))), ('target_items', list(set(tgt.iid)))]:
        pd.DataFrame(dict(raw_id=values, id=range(len(values)))).to_csv(root / f'data/id_maps/{label}.csv', index=False)
    prep = DataPreprocessingReady(str(root / 'data') + '/', {'1': pair}, '1', [.8, .2])
    src, tgt, overlap = prep.mapper(src, tgt)
    original_sample = random.sample
    try:
        random.sample = lambda population, k: original_sample(list(population), k)
        train_src, train_tgt, test, meta, test_ids = prep.split(src, tgt, overlap)
    finally:
        random.sample = original_sample
    assert not set(train_tgt.uid) & set(test.uid)
    assert set(test.uid) <= set(train_src.uid)
    prep.save(train_src, train_tgt, test, meta, test_ids)
    save(root / 'data/split_manifest.json', dict(seed=2020, pythonhashseed='2020', task=name,
         beta=.2, overlap_users=overlap, cold_users=len(test_ids), source_interactions=len(src),
         target_train_interactions=len(train_tgt), test_interactions=len(test), mapping_interactions=len(meta)))
    return root


def step(root, label, script, args=()):
    marker = root / 'logs' / f'{label}.done.json'
    if marker.exists():
        return
    free = shutil.disk_usage(BASE).free
    if free < 512 * 1024**2:
        raise RuntimeError(f'Less than 512 MiB free before {label}: {free} bytes; stopping without deleting data')
    save(OUT / 'status.json', dict(task=root.name, stage=label, state='running', updated=time.time()))
    print('START', root.name, label, flush=True)
    start = time.monotonic()
    env = dict(os.environ, CDR_RUN_ROOT=str(root), PYTHONHASHSEED='2020', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0')
    with (root / 'logs' / f'{label}.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(BASE / 'scripts' / script), *args],
                       cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    save(marker, dict(seconds=time.monotonic() - start))
    print('DONE', root.name, label, flush=True)


def music_data():
    mid = BASE / 'data/mid/CDs_and_Vinyl.csv'
    if mid.exists():
        return
    archive = BASE / 'data/raw/reviews_CDs_and_Vinyl_5.json.gz'
    if not archive.with_suffix('.gz.complete').exists():
        download(archive.name)
    # Parsing through EOF also verifies the gzip CRC.
    partial = mid.with_suffix('.partial')
    with gzip.open(archive, 'rt') as source, partial.open('w') as dest:
        writer = csv.writer(dest)
        writer.writerow(['uid', 'iid', 'y'])
        for line in source:
            row = json.loads(line)
            writer.writerow([row['reviewerID'], row['asin'], row['overall']])
    partial.replace(mid)


def compare(root):
    predictions = {name: np.load(root / f'results/{name}_predictions.npz')
                   for name in ['wb', 'bb', 'list_encoder']}
    ref = predictions['wb']
    result = dict(task=root.name, seed=2020, epochs=10, beta=.2, metrics={})
    for name, data in predictions.items():
        for key in ['uid', 'iid', 'y']:
            np.testing.assert_array_equal(ref[key], data[key])
        error = data['prediction'] - data['y']
        assert np.isfinite(error).all()
        result['metrics'][name] = dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())))
    save(root / 'results/comparison.json', result)
    print('COMPARISON', json.dumps(result), flush=True)


def main():
    assert os.environ.get('PYTHONHASHSEED') == '2020'
    OUT.mkdir(exist_ok=True)
    # Fixed settings only; no test-driven tuning, no mechanistic experiments.
    for name, target in [('movie_to_book', 'Books'), ('movie_to_music', 'CDs_and_Vinyl')]:
        save(OUT / 'status.json', dict(task=name, stage='prepare', state='running', updated=time.time()))
        if target == 'CDs_and_Vinyl':
            music_data()
        root = prepare(name, target)
        step(root, 'base', 'experiment.py', ['train'])
        step(root, 'api', 'export_api.py')
        # Get the new method's fixed result before the more expensive references.
        step(root, 'list_encoder', 'run_list_encoder.py')
        step(root, 'extract', 'extract_surrogate.py')
        shutil.copy2(root / 'checkpoints/bb_learner/surrogate_source.pt', root / 'checkpoints/surrogate_source.pt')
        step(root, 'wb_invert', 'experiment.py', ['invert', '--source', 'wb'])
        step(root, 'wb_bridge', 'experiment.py', ['bridge', '--source', 'wb'])
        step(root, 'bb_invert', 'experiment.py', ['invert', '--source', 'bb'])
        step(root, 'bb_bridge', 'experiment.py', ['bridge', '--source', 'bb'])
        compare(root)
    save(OUT / 'status.json', dict(state='complete', updated=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status = json.loads((OUT / 'status.json').read_text()) if (OUT / 'status.json').exists() else {}
        status.update(state='failed', error=repr(exc), updated=time.time())
        save(OUT / 'status.json', status)
        raise
