#!/bin/bash
#SBATCH --job-name=paper-list-dnn
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/paper_list_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/paper_list_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 2
study_root="$PWD/runs/paper_list_dnn_${SLURM_JOB_ID}"
python -u scripts/paper_list_dnn.py prepare --root "$study_root"
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/paper_list_dnn.py selftest --root "$study_root"
tasks=(book_movie book_music music_movie_clean)
worker() {
    local lane="$1"
    local index task seed beta sub
    for ((index=lane; index<9; index+=2)); do
        task="${tasks[$((index/3))]}"
        seed=$((2020+index%3))
        sub="$study_root/$task/seed_$seed"
        CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/paper_list_dnn.py source --root "$study_root" --task "$task" --seed "$seed" > "$sub/queries/DNN/source.log" 2>&1
        for beta in 20 50 80; do
            CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/paper_list_dnn.py train --root "$study_root" --task "$task" --seed "$seed" --beta "$beta" > "$study_root/$task/beta_$beta/seed_$seed/train.log" 2>&1
        done
    done
}
pids=()
for lane in 0 1; do worker "$lane" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
python -u scripts/paper_list_dnn.py evaluate --root "$study_root"
