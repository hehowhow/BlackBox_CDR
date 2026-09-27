#!/bin/bash
#SBATCH --job-name=list-story-pair
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/list_story_pair_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/list_story_pair_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
: "${SLURM_JOB_ID:?Slurm job required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
IFS=',' read -r -a story_devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#story_devices[@]}" -eq 2

# The account permits one running job, at most two GPUs. The original per-track
# scripts run unchanged inside this allocation, each seeing one allocated GPU.
CUDA_VISIBLE_DEVICES="${story_devices[0]}" bash scripts/slurm_list_ib_story.sh \
    > "logs/list_ib_story_${SLURM_JOB_ID}.out" 2> "logs/list_ib_story_${SLURM_JOB_ID}.err" &
ib_pid=$!
CUDA_VISIBLE_DEVICES="${story_devices[1]}" bash scripts/slurm_list_acquisition_story.sh \
    > "logs/list_acquisition_story_${SLURM_JOB_ID}.out" 2> "logs/list_acquisition_story_${SLURM_JOB_ID}.err" &
acquisition_pid=$!
ib_status=0
acquisition_status=0
wait "$ib_pid" || ib_status=$?
wait "$acquisition_pid" || acquisition_status=$?
printf 'IB exit=%s; acquisition exit=%s\n' "$ib_status" "$acquisition_status"
test "$ib_status" -eq 0
test "$acquisition_status" -eq 0
