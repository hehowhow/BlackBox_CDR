#!/bin/bash
#SBATCH --job-name=mean-leak-audit
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/mean_audit_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/mean_audit_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for case in 0 1 2 3; do
    seed=2020
    if [ "$case" = 3 ]; then seed=2021; fi
    PYTHONHASHSEED="$seed" python -u scripts/audit_mean_leakage.py --case "$case"
done
