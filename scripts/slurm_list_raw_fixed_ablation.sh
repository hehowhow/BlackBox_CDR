#!/bin/bash
#SBATCH --job-name=list-raw-ablation
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_raw_fixed_ablation_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_raw_fixed_ablation_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${SLURM_JOB_ID:?Slurm required}"
: "${CUDA_VISIBLE_DEVICES:?Allocated GPUs required}"
IFS=',' read -r -a fixed_devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#fixed_devices[@]}" -eq 2
fixed_root="$PWD/runs/list_raw_fixed_ablation_${SLURM_JOB_ID}"
python -u scripts/list_raw_fixed_ablation.py prepare --root "$fixed_root"
python -m pip freeze > "$fixed_root/environment.txt"
echo "Allocated GPUs: ${fixed_devices[*]}; use only idle allocated devices" > "$fixed_root/gpu_policy.txt"
CUDA_VISIBLE_DEVICES="${fixed_devices[0]}" python -u scripts/list_raw_fixed_ablation.py worker --root "$fixed_root" --index 0 > "$fixed_root/worker_0.log" 2>&1 &
fixed_first=$!
CUDA_VISIBLE_DEVICES="${fixed_devices[1]}" python -u scripts/list_raw_fixed_ablation.py worker --root "$fixed_root" --index 1 > "$fixed_root/worker_1.log" 2>&1 &
fixed_second=$!
fixed_a=0; fixed_b=0
wait "$fixed_first" || fixed_a=$?
wait "$fixed_second" || fixed_b=$?
test "$fixed_a" -eq 0
test "$fixed_b" -eq 0
python -u scripts/list_raw_fixed_ablation.py evaluate --root "$fixed_root"
python -u scripts/verify_list_raw_fixed_ablation.py --root "$fixed_root"
