#!/bin/bash
#SBATCH --job-name=pfcdr-backbones
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/backbones_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/backbones_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 2
study_root="$PWD/runs/pfcdr_backbones_${SLURM_JOB_ID}"
python -u scripts/pfcdr_backbone_study.py prepare --root "$study_root"
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/pfcdr_backbone_study.py selftest --root "$study_root"
names=(MF GMF DNN)
pids=()
worker() {
    local lane="$1"
    local index
    for ((index=lane; index<3; index+=2)); do
        CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/pfcdr_backbone_study.py run --root "$study_root" --backbone "${names[$index]}" > "$study_root/${names[$index]}.log" 2>&1
    done
}
for lane in 0 1; do
    worker "$lane" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
test "$status" -eq 0
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/pfcdr_backbone_verify.py --root "$study_root"
python -u scripts/pfcdr_backbone_study.py evaluate --root "$study_root"
