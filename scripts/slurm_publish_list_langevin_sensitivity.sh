#!/bin/bash
#SBATCH --job-name=list-langevin-report
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_langevin_report_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_langevin_report_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
exec /home/hehao32/miniconda3/envs/blackbox-cdr/bin/python scripts/publish_list_langevin_sensitivity.py --job "$1"
