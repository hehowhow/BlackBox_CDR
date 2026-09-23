#!/bin/bash
#SBATCH --job-name=gmf-match-control
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/gmf_match_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/gmf_match_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
python -u scripts/pfcdr_gmf_matching_control.py run --root "$1"
python -u scripts/pfcdr_gmf_matching_control.py evaluate --root "$1"
