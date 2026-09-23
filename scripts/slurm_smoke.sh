#!/bin/bash
#SBATCH --job-name=blackbox-pfcdr-smoke
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/smoke_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/smoke_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export PYTHONHASHSEED=2020 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# Preserve Slurm's allocation; never override it with a physical GPU index.
: "${SLURM_JOB_ID:?Submit this script using sbatch}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign a GPU}"
export CDR_RUN_ROOT="$PWD/runs/slurm_smoke_${SLURM_JOB_ID}"

python - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import torch

repo = Path.cwd()
root = Path(os.environ['CDR_RUN_ROOT'])
assert (repo / 'data/split_manifest.json').exists(), 'Prepare the full dataset first'
root.mkdir(exist_ok=False)
for name in ('configs', 'checkpoints', 'results', 'logs'):
    (root / name).mkdir()
(root / 'data').symlink_to(repo / 'data', target_is_directory=True)
cfg = json.loads((repo / 'configs/first_round.json').read_text())
cfg.update(epoch=1, inversion_iter=2)
(root / 'configs/first_round.json').write_text(json.dumps(cfg, indent=2) + '\n')
assert torch.cuda.is_available(), 'CUDA is required'
assert torch.cuda.device_count() == 1, 'Expect exactly one allocated GPU'
a = torch.randn(32, 32, device='cuda')
assert torch.isfinite(a @ a.T).all()
info = dict(job_id=os.environ['SLURM_JOB_ID'],
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
            torch=torch.__version__, cuda=torch.version.cuda,
            repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            config=cfg, purpose='Full-data functional smoke test, not paper metric reproduction')
(root / 'environment.json').write_text(json.dumps(info, indent=2) + '\n')
print(json.dumps(info, indent=2), flush=True)
PY

for stage in train invert bridge; do
    echo "Starting $stage at $(date --iso-8601=seconds)"
    python -u scripts/experiment.py "$stage" --source wb > "$CDR_RUN_ROOT/logs/wb_${stage}.log" 2>&1
    echo "Completed $stage at $(date --iso-8601=seconds)"
done

python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

root = Path(os.environ['CDR_RUN_ROOT'])
base = json.loads((root / 'results/base_training.json').read_text())
bridge = json.loads((root / 'results/wb_training.json').read_text())
assert len(base) == 2 and {row['domain'] for row in base} == {'src', 'tgt'}
assert all(row['epoch'] == 1 for row in base)
assert len(bridge) == 1 and bridge[0]['epoch'] == 1
data = np.load(root / 'results/wb_predictions.npz')
manifest = json.loads((root / 'data/split_manifest.json').read_text())
assert len(data['prediction']) == manifest['test_interactions']
assert np.isfinite(data['prediction']).all()
error = data['prediction'] - data['y']
mae = float(np.abs(error).mean())
rmse = float(np.sqrt(np.square(error).mean()))
assert np.isclose(mae, bridge[0]['mae'], rtol=1e-5)
assert np.isclose(rmse, bridge[0]['rmse'], rtol=1e-5)
summary = dict(status='PASS', job_id=os.environ['SLURM_JOB_ID'],
               task='Book->Movie PFCDR-WB', epochs=1, inversion_iterations=2,
               test_ratings=len(error), test_users=len(np.unique(data['uid'])),
               mae=mae, rmse=rmse, full_dataset=True)
(root / 'smoke_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('SMOKE_TEST_PASS', json.dumps(summary), flush=True)
PY
