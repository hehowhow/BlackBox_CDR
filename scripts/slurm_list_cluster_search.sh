#!/bin/bash
#SBATCH --job-name=list-cluster-grid
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_cluster_search_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_cluster_search_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 2
study_root="$PWD/runs/list_cluster_search_${SLURM_JOB_ID}"
python -u scripts/list_cluster_search.py prepare --root "$study_root"
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/list_cluster_search.py selftest --root "$study_root"
worker() {
    local lane="$1" beta count seed
    local betas=(20 80)
    beta="${betas[$lane]}"
    for count in 1 2 3 4 8 16; do
        for seed in 2020 2021 2022; do
            python -u scripts/list_cluster_search.py fit_groups --root "$study_root" --count "$count" --beta "$beta" --seed "$seed"
            CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/list_cluster_search.py train --root "$study_root" --count "$count" --beta "$beta" --seed "$seed" > "$study_root/c_$count/cooccurrence/beta_$beta/seed_$seed/residual_real/select_train.log" 2>&1
        done
    done
    python -u scripts/list_cluster_search.py choose --root "$study_root" --beta "$beta"
    CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/list_cluster_search.py finals --root "$study_root" --beta "$beta"
}
pids=()
for lane in 0 1; do worker "$lane" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
python -u scripts/list_cluster_search.py evaluate --root "$study_root"
