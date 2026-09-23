#!/bin/bash
#SBATCH --job-name=cdr-residual-train
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/residual_train_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/residual_train_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
IFS=',' read -r -a allocated_gpus <<< "$CUDA_VISIBLE_DEVICES"
test "${#allocated_gpus[@]}" -eq 2
study_root="$1"
worker() {
    local lane="$1"
    local device="$2"
    for ((index=lane; index<9; index+=2)); do
        echo "START index=$index gpu=$device $(date --iso-8601=seconds)"
        CUDA_VISIBLE_DEVICES="$device" python -u scripts/residual_experiment.py source --root "$study_root" --index "$index" > "$study_root/train_${index}.log" 2>&1
        CUDA_VISIBLE_DEVICES="$device" python -u scripts/residual_experiment.py learn --root "$study_root" --index "$index" >> "$study_root/train_${index}.log" 2>&1
        echo "DONE index=$index $(date --iso-8601=seconds)"
    done
}
worker 0 "${allocated_gpus[0]}" &
lane_zero=$!
worker 1 "${allocated_gpus[1]}" &
lane_one=$!
wait "$lane_zero"
wait "$lane_one"
python -u scripts/residual_summary.py --root "$study_root"
