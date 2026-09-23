#!/bin/bash
#SBATCH --job-name=list-dnn-transfer
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_transfer_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_transfer_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 2
study_root="$PWD/runs/list_source_transfer_${SLURM_JOB_ID}"
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/list_source_transfer.py prepare --root "$study_root"
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/list_source_transfer.py selftest --root "$study_root"
tasks=(movie_book movie_music_clean)
worker() {
    local lane="$1"
    local index task seed name sub
    for ((index=lane; index<6; index+=2)); do
        task="${tasks[$((index/3))]}"
        seed=$((2020+index%3))
        sub="$study_root/$task/seed_$seed"
        for name in MF DNN; do
            CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/list_source_transfer.py source --root "$study_root" --task "$task" --seed "$seed" --backbone "$name" > "$sub/queries/$name/source.log" 2>&1
        done
        CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/list_source_transfer.py shared --root "$study_root" --task "$task" --seed "$seed" > "$sub/shared.log" 2>&1
        for name in MF DNN; do
            CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/list_source_transfer.py train --root "$study_root" --task "$task" --seed "$seed" --backbone "$name" > "$sub/students/$name/train.log" 2>&1
        done
        CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/list_source_transfer.py verify --root "$study_root" --task "$task" --seed "$seed" > "$sub/verify.log" 2>&1
    done
}
pids=()
for lane in 0 1; do
    worker "$lane" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
test "$status" -eq 0
python -u scripts/list_source_transfer.py evaluate --root "$study_root"
