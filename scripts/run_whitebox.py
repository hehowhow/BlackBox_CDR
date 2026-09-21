"""Sequential, fail-fast launch of the first two experimental groups."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
env = dict(os.environ, PYTHONHASHSEED='2020', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4')
for stage in ['train', 'invert', 'bridge']:
    with (root / 'logs' / f'wb_{stage}.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(root / 'scripts/experiment.py'), stage],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed', stage, flush=True)
