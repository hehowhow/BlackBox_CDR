#!/bin/bash
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=2021
export LIST_EXECUTION_MODE=direct CUDA_DEVICE_ORDER=PCI_BUS_ID
unset SLURM_JOB_ID
reconstruction_root="${1:?Prepared study directory required}"
reconstruction_first_gpu="${2:?First GPU UUID required}"
reconstruction_second_gpu="${3:?Second GPU UUID required}"
test "$reconstruction_first_gpu" != "$reconstruction_second_gpu"
python -u scripts/list_shared_reconstruction_study.py check --root "$reconstruction_root"
python -m pip freeze > "$reconstruction_root/environment.txt"
CUDA_VISIBLE_DEVICES="$reconstruction_first_gpu" python -u scripts/list_shared_reconstruction_study.py worker --root "$reconstruction_root" --index 0 > "$reconstruction_root/worker_0.log" 2>&1 &
reconstruction_first=$!
CUDA_VISIBLE_DEVICES="$reconstruction_second_gpu" python -u scripts/list_shared_reconstruction_study.py worker --root "$reconstruction_root" --index 1 > "$reconstruction_root/worker_1.log" 2>&1 &
reconstruction_second=$!
printf '%s\n' "$reconstruction_first" "$reconstruction_second" > "$reconstruction_root/worker_pids.txt"
reconstruction_a=0; reconstruction_b=0
wait "$reconstruction_first" || reconstruction_a=$?
wait "$reconstruction_second" || reconstruction_b=$?
test "$reconstruction_a" -eq 0
test "$reconstruction_b" -eq 0
CUDA_VISIBLE_DEVICES='' python -u scripts/list_shared_reconstruction_study.py evaluate --root "$reconstruction_root"
CUDA_VISIBLE_DEVICES='' python -u scripts/verify_list_shared_reconstruction.py --root "$reconstruction_root"
