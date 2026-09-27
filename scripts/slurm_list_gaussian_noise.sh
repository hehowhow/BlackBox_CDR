#!/bin/bash
#SBATCH --job-name=list-gaussian-noise
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_gaussian_noise_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_gaussian_noise_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${SLURM_JOB_ID:?Slurm job required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
IFS=',' read -r -a noise_devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#noise_devices[@]}" -eq 2
study_root="$PWD/runs/list_gaussian_noise_${SLURM_JOB_ID}"
python -u scripts/list_gaussian_noise.py prepare --root "$study_root"
CUDA_VISIBLE_DEVICES="${noise_devices[0]}" python -u scripts/list_gaussian_noise.py selftest --root "$study_root"
CUDA_VISIBLE_DEVICES="${noise_devices[0]}" python -u scripts/list_gaussian_noise.py worker --root "$study_root" --worker 0 > "$study_root/worker_0.log" 2>&1 &
first_pid=$!
CUDA_VISIBLE_DEVICES="${noise_devices[1]}" python -u scripts/list_gaussian_noise.py worker --root "$study_root" --worker 1 > "$study_root/worker_1.log" 2>&1 &
second_pid=$!
first_status=0
second_status=0
wait "$first_pid" || first_status=$?
wait "$second_pid" || second_status=$?
test "$first_status" -eq 0
test "$second_status" -eq 0
python -u scripts/list_gaussian_noise.py freeze --root "$study_root"
python -u scripts/list_gaussian_noise.py evaluate --root "$study_root"
python -u scripts/verify_list_gaussian_noise.py --root "$study_root"
