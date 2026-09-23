#!/bin/bash
#SBATCH --job-name=list-target-heads
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_heads_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_heads_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=00:45:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 2
study_root="$PWD/runs/list_target_heads_${SLURM_JOB_ID}"
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/list_target_heads.py prepare --root "$study_root"
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/list_target_heads.py selftest --root "$study_root"
heads=(DOT GMF DNN MLP DOT_MLP)
worker() {
    local lane="$1"
    local index seed name
    for ((index=lane; index<15; index+=2)); do
        seed=$((2020+index/5))
        name="${heads[$((index%5))]}"
        CUDA_VISIBLE_DEVICES="${devices[$lane]}" python -u scripts/list_target_heads.py train --root "$study_root" --seed "$seed" --head "$name" > "$study_root/seed_$seed/students/$name/train.log" 2>&1
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
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u scripts/list_target_heads.py verify --root "$study_root"
python -u scripts/list_target_heads.py evaluate --root "$study_root"
