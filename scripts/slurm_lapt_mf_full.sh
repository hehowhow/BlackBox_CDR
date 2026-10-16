#!/bin/bash
#SBATCH --job-name=lapt-mf-full
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/lapt_mf_full_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/lapt_mf_full_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
set -euo pipefail
cd /home/hehao32/BlackBox_CDR
study_root="$1"
study_python=/home/hehao32/miniconda3/envs/blackbox-cdr/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=2020
: "${SLURM_JOB_ID:?Slurm required}"
: "${CUDA_VISIBLE_DEVICES:?Allocated GPUs required}"
: "${SLURM_JOB_GPUS:?Global allocated GRES indices required}"
# This host uses AutoDetect=off and File=/dev/nvidia[0-7]. GRES indices
# therefore name device MINOR numbers, which differ from CUDA/NVML order.
# Resolve allocated device files to UUIDs instead of using the numeric mask.
study_mask=$("$study_python" - "$study_root" <<'PY'
import json,os,subprocess,sys,xml.etree.ElementTree as E
from pathlib import Path
raw=os.environ['SLURM_JOB_GPUS']
indices=[]
for field in raw.split(','):
    if '-' in field:
        lo,hi=map(int,field.split('-'));indices.extend(range(lo,hi+1))
    else:indices.append(int(field))
assert len(indices)==2
xml=E.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
devices={int(g.findtext('minor_number')):dict(nvml_index=i,minor=int(g.findtext('minor_number')),uuid=g.findtext('uuid'))
         for i,g in enumerate(xml.findall('gpu'))}
selected=[devices[i] for i in indices]
record=dict(job=os.environ['SLURM_JOB_ID'],slurm_job_gpus=raw,
            original_cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            gres_config='AutoDetect=off; File=/dev/nvidia[0-7]',devices=selected)
(Path(sys.argv[1])/'gpu_allocation.json').write_text(json.dumps(record,indent=2))
print(','.join(d['uuid'] for d in selected))
PY
)
IFS=',' read -r -a study_devices <<< "$study_mask"
test "${#study_devices[@]}" -eq 2
printf 'Slurm job %s resolved allocated devices %s\n' "$SLURM_JOB_ID" "$study_mask"
CUDA_VISIBLE_DEVICES="${study_devices[0]}" "$study_python" -u scripts/lapt_mf_full.py worker --root "$study_root" --index 0 > "$study_root/worker_0.log" 2>&1 &
first_pid=$!
CUDA_VISIBLE_DEVICES="${study_devices[1]}" "$study_python" -u scripts/lapt_mf_full.py worker --root "$study_root" --index 1 > "$study_root/worker_1.log" 2>&1 &
second_pid=$!
trap 'kill "$first_pid" "$second_pid" 2>/dev/null || true' TERM INT
first_status=0
second_status=0
wait "$first_pid" || first_status=$?
wait "$second_pid" || second_status=$?
test "$first_status" -eq 0
test "$second_status" -eq 0
"$study_python" -u scripts/lapt_mf_full.py evaluate --root "$study_root"
