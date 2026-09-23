#!/bin/bash
#SBATCH --job-name=blackbox-data
#SBATCH --partition=LocalQ
#SBATCH --output=/home/hehao32/BlackBox_CDR/logs/prepare_%j.out
#SBATCH --error=/home/hehao32/BlackBox_CDR/logs/prepare_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
export PYTHONHASHSEED=2020 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python scripts/prepare_data.py
python - <<'PY'
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'scripts')
from run_extra_tasks import music_data
music_data()

root = Path.cwd()
records = []
for domain in ('Books', 'Movies_and_TV', 'CDs_and_Vinyl'):
    archive = root / 'data/raw' / f'reviews_{domain}_5.json.gz'
    mid = root / 'data/mid' / f'{domain}.csv'
    with archive.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    with mid.open('rb') as stream:
        rows = sum(1 for _ in stream) - 1
    records.append(dict(domain=domain, bytes=archive.stat().st_size,
                        sha256=checksum, interactions=rows,
                        source='https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/' + archive.name))
expected = dict(Books=8898041, Movies_and_TV=1697533, CDs_and_Vinyl=1097592)
assert all(row['interactions'] == expected[row['domain']] for row in records), records
manifest = dict(dataset='Amazon Reviews 2014 5-core',
                upstream='https://github.com/walcheng/PFCDR', archives=records,
                validation='HTTP range and byte counts; parsed gzip through EOF (CRC); expected record counts; local SHA256')
(root / 'data/download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('DATA_PREPARATION_PASS', json.dumps(manifest), flush=True)
PY
