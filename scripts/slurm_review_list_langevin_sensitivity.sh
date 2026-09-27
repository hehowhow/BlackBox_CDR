#!/bin/bash
#SBATCH --job-name=list-langevin-recheck
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_langevin_recheck_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_langevin_recheck_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python_bin=/home/hehao32/miniconda3/envs/blackbox-cdr/bin/python
"$python_bin" scripts/verify_list_langevin_sensitivity_review.py --root runs/list_langevin_sensitivity_8968
"$python_bin" scripts/publish_list_langevin_sensitivity.py --job 8968
