#!/bin/bash
#SBATCH --job-name=list-memory-story
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_memory_story_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_memory_story_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
study_root="$PWD/runs/list_memory_story_${SLURM_JOB_ID}"
python -u scripts/list_memory_story.py prepare --root "$study_root"
python -u scripts/list_memory_story.py selftest --root "$study_root"
for variant in pair20 pair50; do
    for beta in 20 80; do
        for seed in 2020 2021 2022; do
            python -u scripts/list_memory_story.py train --root "$study_root" --variant "$variant" --beta "$beta" --seed "$seed" > "$study_root/$variant/beta_$beta/seed_$seed/residual_real/train.log" 2>&1
        done
    done
done
python -u scripts/list_memory_story.py evaluate --root "$study_root"
