#!/bin/bash
#SBATCH --job-name=list-source-backbones
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_source_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_source_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
study_root="$PWD/runs/list_source_backbones_${SLURM_JOB_ID}"
python -u scripts/list_source_backbones.py prepare --root "$study_root"
python -u scripts/list_source_backbones.py selftest --root "$study_root"
for name in MF GMF DNN; do
    python -u scripts/list_source_backbones.py export --root "$study_root" --backbone "$name" > "$study_root/queries/$name/export.log" 2>&1
done
python -u scripts/list_source_backbones.py shared --root "$study_root"
for name in MF GMF DNN; do
    python -u scripts/list_source_backbones.py train --root "$study_root" --backbone "$name" > "$study_root/students/$name/train.log" 2>&1
done
python -u scripts/list_source_backbones.py verify --root "$study_root"
python -u scripts/list_source_backbones.py evaluate --root "$study_root"
