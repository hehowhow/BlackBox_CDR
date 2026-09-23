#!/bin/bash
#SBATCH --job-name=list-weighting
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_weighting_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_weighting_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 2
study_root="$PWD/runs/list_weighting_${SLURM_JOB_ID}"
python -u scripts/list_weighting.py prepare --root "$study_root"
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/list_weighting.py selftest --root "$study_root"
worker() {
    local lane="$1" variant beta seed arm
    local variants=(inverse_degree target_attention)
    variant="${variants[$lane]}"
    for beta in 20 80; do
        for seed in 2020 2021 2022; do
            for arm in real shuffled; do
                CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/list_weighting.py train --root "$study_root" --variant "$variant" --beta "$beta" --seed "$seed" --arm "$arm" > "$study_root/$variant/beta_$beta/seed_$seed/residual_$arm/train.log" 2>&1
            done
        done
    done
}
pids=()
for lane in 0 1; do worker "$lane" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
python -u scripts/list_weighting.py evaluate --root "$study_root"
