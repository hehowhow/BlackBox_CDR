"""Confirm cross-domain repeated reviews directly in both original gzip inputs."""
import gzip
import hashlib
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'results/list_audit'
target_users = {}
all_users = set()
for name, root in [('2020', BASE/'runs/movie_to_music'), ('2021', BASE/'runs/fresh_movie_music_seed2021/list')]:
    ids = pd.read_csv(root/'data/id_maps/target_users.csv', dtype={'raw_id':str})
    test = pd.read_csv(root/'data/ready/_8_2/tgt_CDs_and_Vinyl_src_Movies_and_TV/test.csv', header=None, usecols=[0])
    users = set(ids.loc[ids.id.isin(test[0].unique()), 'raw_id'])
    target_users[name] = users
    all_users |= users
records = {}
for domain in ['Movies_and_TV', 'CDs_and_Vinyl']:
    selected = {}
    with gzip.open(BASE/f'data/raw/reviews_{domain}_5.json.gz', 'rt') as source:
        for line in source:
            row = json.loads(line)
            if row['reviewerID'] not in all_users:
                continue
            key = (row['reviewerID'], row['asin'])
            content = [row.get(k) for k in ['overall', 'unixReviewTime', 'reviewText', 'summary']]
            selected[key] = hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).digest()
    records[domain] = selected
    print('READ_RAW', domain, len(selected), flush=True)
a, b = records['Movies_and_TV'], records['CDs_and_Vinyl']
overlap = set(a) & set(b)
result = {}
for seed, users in target_users.items():
    keys = [k for k in overlap if k[0] in users]
    result[seed] = dict(same_user_item=len(keys),
                       identical_rating_timestamp_review_text_and_summary=sum(a[k] == b[k] for k in keys))
(OUT/'raw_review_duplicates.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
