#!/bin/bash
#SBATCH --job-name=cdr-residual-eval
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/residual_summary_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/residual_summary_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -u scripts/residual_summary.py --root "$1"
