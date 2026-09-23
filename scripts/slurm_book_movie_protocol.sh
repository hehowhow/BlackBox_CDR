#!/bin/bash
#SBATCH --job-name=book-movie-protocol
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/protocol_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/protocol_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
run_root="$PWD/runs/book_movie_protocol_${SLURM_JOB_ID}"
for stage in prepare validate test; do
    python -u scripts/book_movie_protocol.py "$stage" --root "$run_root"
done
