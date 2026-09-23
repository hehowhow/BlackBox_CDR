#!/bin/bash
#SBATCH --job-name=small-cdr-compare
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/small_compare_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/small_compare_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
study_root="$PWD/runs/small_task_comparison_${SLURM_JOB_ID}"
python -u scripts/small_task_comparison.py prepare --root "$study_root"
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/small_task_comparison.py selftest --root "$study_root"
worker() {
    local lane="$1" index seed name beta arm sub
    local names=(MF DNN)
    name="${names[$lane]}"
    for seed in 2020 2021 2022; do
        sub="$study_root/sources/seed_$seed/$name"
        CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/small_task_comparison.py source --root "$study_root" --seed "$seed" --backbone "$name" > "$sub/train.log" 2>&1
        for beta in 20 80; do
            sub="$study_root/beta_$beta/seed_$seed"
            CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/small_task_comparison.py baseline --root "$study_root" --seed "$seed" --beta "$beta" --backbone "$name" > "$sub/$name/train.log" 2>&1
            if [ "$lane" -eq 0 ]; then arm=real; else arm=shuffled; fi
            CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/small_task_comparison.py residual --root "$study_root" --seed "$seed" --beta "$beta" --arm "$arm" > "$sub/residual_$arm/train.log" 2>&1
        done
    done
}
pids=()
for lane in 0 1; do worker "$lane" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
python -u scripts/small_task_comparison.py evaluate --root "$study_root"
