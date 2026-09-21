"""Replay unchanged List training with shuffled test labels and guarded reads."""
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parents[1]
ORIGINAL = BASE / 'runs/fresh_movie_music_seed2021/list'
OUT = BASE / 'results/list_audit'
REL = Path('data/ready/_8_2/tgt_CDs_and_Vinyl_src_Movies_and_TV')


def child(root):
    os.environ['CDR_RUN_ROOT'] = str(root)
    torch.use_deterministic_algorithms(True)
    checkpoint_saved = False
    inputs = [(root/'configs/first_round.json').resolve(), (root/'queries/topk_scores.npz').resolve(),
              (root/REL/'train_tgt.csv').resolve(), root/REL/'test.csv']
    reads = set()
    original_save = torch.save
    def guarded_save(obj, path, *args, **kwargs):
        nonlocal checkpoint_saved
        result = original_save(obj, path, *args, **kwargs)
        if Path(path) == root/'checkpoints/list_encoder.pt':
            checkpoint_saved = True
        return result
    def no_load(*args, **kwargs):
        raise RuntimeError('Audit: pretrained checkpoint load prohibited')
    torch.save = guarded_save
    torch.load = no_load
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        flags = args[2]
        if isinstance(args[1], str) and any(c in args[1] for c in ['w', 'a', 'x']):
            return
        reading = isinstance(flags, int) and (flags & os.O_ACCMODE) != os.O_WRONLY
        if not reading or not path.is_relative_to(BASE):
            return
        if path.suffix in ['.py', '.pyc']:
            return
        if path not in inputs:
            raise RuntimeError(f'Audit: unexpected project input {path}')
        if path == root/REL/'test.csv' and not checkpoint_saved:
            raise RuntimeError('Audit: test labels accessed before checkpoint saved')
        reads.add(str(path))
    sys.addaudithook(audit)
    runpy.run_path(str(BASE/'scripts/run_list_encoder.py'), run_name='__main__')
    (root/'read_audit.json').write_text(json.dumps(dict(reads=sorted(reads),
                      checkpoint_saved_before_test_read=checkpoint_saved, torch_load_calls=0), indent=2))


def parent():
    OUT.mkdir(exist_ok=True)
    roots = []
    for shuffle in [False, True]:
        root = Path(tempfile.mkdtemp(prefix='replay_shuffled_' if shuffle else 'replay_control_', dir=OUT))
        roots.append(root)
        for folder in ['configs', 'queries', 'checkpoints', 'results', str(REL)]:
            (root/folder).mkdir(parents=True, exist_ok=True)
        for rel in [Path('configs/first_round.json'), Path('queries/topk_scores.npz'), REL/'train_tgt.csv']:
            (root/rel).symlink_to(ORIGINAL/rel)
        test = pd.read_csv(ORIGINAL/REL/'test.csv', header=None)
        if shuffle:
            test[2] = np.random.default_rng(996).permutation(test[2].to_numpy())
        test.to_csv(root/REL/'test.csv', index=False, header=False)
        with (root/'training.log').open('w') as log:
            subprocess.run([sys.executable, '-u', __file__, str(root)],
                           env=dict(os.environ, PYTHONHASHSEED='2021', OMP_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    control, root = roots
    a = torch.load(control/'checkpoints/list_encoder.pt', weights_only=True, map_location='cpu')
    b = torch.load(root/'checkpoints/list_encoder.pt', weights_only=True, map_location='cpu')
    diffs = {k: float((a['state_dict'][k]-b['state_dict'][k]).abs().max()) for k in a['state_dict']}
    old = np.load(control/'results/list_encoder_predictions.npz')
    new = np.load(root/'results/list_encoder_predictions.npz')
    delta = float(np.max(np.abs(old['prediction']-new['prediction'])))
    result = dict(replay_root=str(root), control_root=str(control), deterministic_audit_retraining=True,
                  input_scope='two fresh students on the same permitted target training data/API outputs; only test labels changed',
                  changed_test_label_rows=int(np.sum(old['y'] != new['y'])),
                  checkpoint_tensor_max_abs_diff=max(diffs.values()), prediction_max_abs_diff=delta,
                  real_label_mae=float(np.abs(new['prediction']-old['y']).mean()),
                  shuffled_label_mae=float(np.abs(new['prediction']-new['y']).mean()),
                  guard=json.loads((root/'read_audit.json').read_text()))
    (OUT/'replay_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    child(Path(sys.argv[1])) if len(sys.argv) > 1 else parent()
