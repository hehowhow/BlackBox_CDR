"""Trusted source-service side: return only Top-K IDs and scores to the learner."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('CDR_RUN_ROOT', Path(__file__).resolve().parents[1]))
import numpy as np
import torch

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
cfg = json.loads((ROOT / 'configs/blackbox.json').read_text())
state = torch.load(ROOT / 'checkpoints/victim_source.pt', weights_only=True, map_location='cuda')
users = state['uid_embedding.weight']
items = state['iid_embedding.weight']
overlap = json.loads((ROOT / 'data/split_manifest.json').read_text())['overlap_users']
ids, scores = [], []
start_time = time.monotonic()
with torch.no_grad():
    for start in range(0, overlap, 128):
        values, indices = torch.topk(users[start:min(start + 128, overlap)] @ items.T, cfg['k'], dim=1)
        ids.append(indices.cpu().numpy().astype(np.int32))
        scores.append(values.cpu().numpy())
        if start % 4096 == 0:
            print('Queries', min(start + 128, overlap), '/', overlap, flush=True)
np.savez_compressed(ROOT / 'queries/topk_scores.npz', user_ids=np.arange(overlap, dtype=np.int32),
                    item_ids=np.concatenate(ids), scores=np.concatenate(scores))
metadata = {'api': cfg['api'], 'k': cfg['k'], 'query_count': overlap,
            'returned_score_count': overlap * cfg['k'], 'num_source_users': len(users),
            'num_source_items': len(items), 'seen_item_filter': False, 'score_clipping': False,
            'seconds': time.monotonic() - start_time}
(ROOT / 'queries/metadata.json').write_text(json.dumps(metadata, indent=2))
print('API_EXPORT_COMPLETE', json.dumps(metadata), flush=True)
