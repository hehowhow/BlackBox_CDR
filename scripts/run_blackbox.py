"""Run the score-only extraction fallback after the common white-box reference."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
env = dict(os.environ, PYTHONHASHSEED='2020', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4')
while not (root / 'checkpoints/victim_source.pt').exists():
    print('Waiting for completed common source model', flush=True)
    time.sleep(30)
steps = [('api', 'export_api.py', []), ('extract', 'extract_surrogate.py', []),
         ('invert', 'experiment.py', ['invert', '--source', 'bb']),
         ('bridge', 'experiment.py', ['bridge', '--source', 'bb'])]
for name, script, args in steps:
    if name == 'invert':
        while not (root / 'results/wb_predictions.npz').exists():
            print('Surrogate ready; waiting for white-box reference', flush=True)
            time.sleep(30)
    with (root / 'logs' / f'bb_{name}.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(root / 'scripts' / script), *args], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if name == 'extract':
        shutil.copy2(root / 'checkpoints/bb_learner/surrogate_source.pt', root / 'checkpoints/surrogate_source.pt')
    print('Completed', name, flush=True)
subprocess.run([sys.executable, str(root / 'scripts/summarize.py')], check=True, env=env)
