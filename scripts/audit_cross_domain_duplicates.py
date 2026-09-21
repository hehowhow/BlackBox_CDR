"""Check source training records against target test records via original IDs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'results/list_audit'
results = []
for root in [BASE/'runs/movie_to_music', BASE/'runs/fresh_movie_music_seed2021/list']:
    data = root/'data/ready/_8_2/tgt_CDs_and_Vinyl_src_Movies_and_TV'
    src = pd.read_csv(data/'train_src.csv', header=None, names=['uid', 'iid', 'source_rating'])
    test = pd.read_csv(data/'test.csv', header=None, names=['uid', 'iid', 'y'])
    source_items = pd.read_csv(root/'data/id_maps/source_items.csv', dtype={'raw_id':str})
    target_items = pd.read_csv(root/'data/id_maps/target_items.csv', dtype={'raw_id':str})
    source_to_target = source_items.raw_id.map(dict(zip(target_items.raw_id, target_items.id)))
    src = src[src.uid.isin(test.uid.unique())].copy()
    src.iid = src.iid.map(dict(zip(source_items.id, source_to_target)))
    src = src.dropna().astype({'iid': 'int64'})
    assert not src.duplicated(['uid','iid']).any()
    joined = test.merge(src, on=['uid','iid'], how='left', sort=False)
    np.testing.assert_array_equal(joined[['uid','iid','y']].to_numpy(), test.to_numpy())
    leak = joined.source_rating.notna().to_numpy()
    equal = leak & (joined.source_rating.to_numpy() == joined.y.to_numpy())
    metrics = {}
    for method in ['list_encoder', 'wb', 'emcdr']:
        path = root/f'results/{method}_predictions.npz'
        if not path.exists() and method == 'wb' and root.name == 'list':
            path = root.parent/'wb/results/wb_predictions.npz'
        if not path.exists():
            continue
        a = np.load(path)
        for key in ['uid','iid','y']:
            np.testing.assert_array_equal(a[key], test[key])
        err = np.abs(a['prediction'] - a['y'])
        metrics[method] = dict(all_mae=float(err.mean()),
                              overlapping_source_record_mae=float(err[leak].mean()),
                              nonoverlap_subset_mae=float(err[~leak].mean()))
    result = dict(root=str(root), test_rows=len(test),
                  source_train_target_test_same_raw_user_item_rows=int(leak.sum()),
                  same_user_item_and_rating_rows=int(equal.sum()),
                  affected_test_users=int(joined.loc[leak,'uid'].nunique()),
                  contaminated_test_fraction=float(leak.mean()),
                  metrics=metrics,
                  note='Subset metrics reuse the original trained model; not a clean retraining result.')
    results.append(result)
protocol = json.loads((BASE/'runs/fresh_movie_music_seed2021/protocol.json').read_text())
hashes = {name: hashlib.sha256((BASE/'scripts'/name).read_bytes()).hexdigest() == expected
          for name, expected in protocol['scripts_sha256'].items()}
report = dict(duplicates=results, fresh_benchmark_code_hashes_match_record=hashes)
(OUT/'cross_domain_duplicates.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
