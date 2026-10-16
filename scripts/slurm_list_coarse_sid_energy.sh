#!/bin/bash
#SBATCH --job-name=list-coarse-energy
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_coarse_sid_energy_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_coarse_sid_energy_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${SLURM_JOB_ID:?Slurm job required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
IFS=',' read -r -a energy_devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#energy_devices[@]}" -ge 1
test "${#energy_devices[@]}" -le 2
study_root="${1:-$PWD/runs/list_coarse_sid_energy_${SLURM_JOB_ID}}"
# Separate single-GPU jobs may share the same atomic task queue. This allows
# useful work when only one GPU is available without exceeding two GPUs total.
(
    flock 9
    if ! test -f "$study_root/tasks.json"; then
        python -u scripts/list_coarse_sid_energy_study.py prepare --root "$study_root"
    fi
    if ! test -f "$study_root/selftest_cpu.json"; then
        python -u scripts/selftest_list_coarse_sid_energy.py > "$study_root/selftest_cpu.json"
    fi
) 9>"${study_root}.launch.lock"
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$study_root/slurm_${SLURM_JOB_ID}_gpus.txt"
worker_pids=()
for device_index in "${!energy_devices[@]}"; do
    worker_index=$((SLURM_JOB_ID * 10 + device_index))
    CUDA_VISIBLE_DEVICES="${energy_devices[$device_index]}" python -u scripts/list_coarse_sid_energy_study.py worker --root "$study_root" --worker "$worker_index" > "$study_root/worker_${worker_index}.log" 2>&1 &
    worker_pids+=("$!")
done
worker_status=0
for worker_pid in "${worker_pids[@]}"; do
    wait "$worker_pid" || worker_status=$?
done
test "$worker_status" -eq 0
# The last worker evaluates; earlier workers release their GPUs immediately.
(
    flock 9
    if ! test -f "$study_root/verification.json" && python - "$study_root" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
tasks = json.loads((root / 'tasks.json').read_text())
for t in tasks:
    out = root / 'models' / t['task'] / f"beta_{t['beta']}" / f"seed_{t['seed']}" / t['variant']
    if not (out / 'sampling_audit.json').is_file():
        sys.exit(1)
    done = json.loads((out / 'complete.json').read_text())
    if not done.get('full_initialization_matches_gaussian'):
        sys.exit(1)
PY
    then
        python -u scripts/list_coarse_sid_energy_study.py evaluate --root "$study_root"
    fi
) 9>"${study_root}.evaluate.lock"
