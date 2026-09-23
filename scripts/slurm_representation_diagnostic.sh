#!/bin/bash
#SBATCH --job-name=cdr-representation
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/representation_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/representation_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign one GPU}"
study_root="$PWD/runs/representation_diagnostic_${SLURM_JOB_ID}"
python -u scripts/representation_diagnostic.py selftest --root "$study_root"
python -u scripts/representation_diagnostic.py prepare --root "$study_root"
for index in 0 1 2 3 4 5; do
    python -u scripts/representation_diagnostic.py train --root "$study_root" --index "$index" --seed 2020 > "$study_root/train_${index}_2020.log" 2>&1
done
python -u scripts/representation_diagnostic.py screen --root "$study_root"
for index in 0 1 2 3 4 5; do
    for seed in 2021 2022; do
        python -u scripts/representation_diagnostic.py train --root "$study_root" --index "$index" --seed "$seed" > "$study_root/train_${index}_${seed}.log" 2>&1
    done
done
python -u scripts/representation_diagnostic.py evaluate --root "$study_root"
