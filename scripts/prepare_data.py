"""Run PFCDR's official split, persisting IDs and checking cold-user isolation."""
import csv
import gzip
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'external/PFCDR'))
from preprocessing import DataPreprocessingReady
import pandas as pd


def main():
    assert os.environ.get('PYTHONHASHSEED') == '2020', 'Launch with PYTHONHASHSEED=2020'
    random.seed(2020)
    mid = ROOT / 'data/mid'
    mid.mkdir(exist_ok=True)
    for domain in ['Books', 'Movies_and_TV']:
        archive = ROOT / 'data/raw' / f'reviews_{domain}_5.json.gz'
        while not archive.with_suffix('.gz.complete').exists():
            print('Waiting for verified download:', domain, flush=True)
            time.sleep(20)
        path = mid / f'{domain}.csv'
        if path.exists():
            continue
        temp = path.with_suffix('.partial')
        count = 0
        with gzip.open(archive, 'rt') as source, temp.open('w') as dest:
            writer = csv.writer(dest)
            writer.writerow(['uid', 'iid', 'y'])
            for line in source:
                record = json.loads(line)
                writer.writerow([record['reviewerID'], record['asin'], record['overall']])
                count += 1
                if count % 1000000 == 0:
                    print(domain, count, flush=True)
        temp.replace(path)
        print(domain, 'parsed', count, flush=True)
    config = json.loads((ROOT / 'external/PFCDR/config.json').read_text())
    prep = DataPreprocessingReady(str(ROOT / 'data') + '/', config['src_tgt_pairs'], '1', [.8, .2])
    src, tgt = prep.read_mid('Books'), prep.read_mid('Movies_and_TV')
    co = set(src.uid) & set(tgt.uid)
    mappings = ROOT / 'data/id_maps'
    mappings.mkdir(exist_ok=True)
    for name, values in [
        ('source_users', list(co) + list(set(src.uid) - co)),
        ('target_users', list(co) + list(set(tgt.uid) - co)),
        ('source_items', list(set(src.iid))), ('target_items', list(set(tgt.iid)))]:
        pd.DataFrame({'raw_id': values, 'id': range(len(values))}).to_csv(mappings / f'{name}.csv', index=False)
    src, tgt, n = prep.mapper(src, tgt)
    # Python >=3.11 no longer accepts a set in random.sample; keep its integer iteration order.
    official_sample = random.sample
    random.sample = lambda population, k: official_sample(list(population), k)
    train_src, train_tgt, test, meta, test_ids = prep.split(src, tgt, n)
    random.sample = official_sample
    assert not set(train_tgt.uid) & set(test.uid)
    assert not set(meta.uid) & set(test.uid)
    assert set(test.uid) <= set(train_src.uid)
    assert len(test_ids) == round(.2 * n)
    assert n == 37388
    for name, actual in [('uid_src', src.uid.nunique()), ('iid_src', src.iid.nunique()),
                         ('uid_tgt', tgt.uid.nunique()), ('iid_tgt', tgt.iid.nunique())]:
        assert config['src_tgt_pairs']['1'][name] == actual, (name, actual)
    prep.save(train_src, train_tgt, test, meta, test_ids)
    manifest = {'seed': 2020, 'pythonhashseed': '2020', 'task': 'Book->Movie', 'beta': .2,
                'overlap_users': n, 'cold_users': len(test_ids),
                'source_interactions': len(train_src), 'target_train_interactions': len(train_tgt),
                'test_interactions': len(test), 'mapping_interactions': len(meta),
                'split_implementation': 'official preprocessing.py with random.sample(set)->list compatibility'}
    (ROOT / 'data/split_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
