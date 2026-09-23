"""Resume official Amazon archives using validated HTTP range requests."""
import concurrent.futures
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'


def download(name):
    path = ROOT / 'data/raw' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = path.with_suffix(path.suffix + '.ranges.json')
    url = BASE + name
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
        total = int(r.headers['Content-Length'])
    if manifest.exists():
        state = json.loads(manifest.read_text())
        assert state['total'] == total
    else:
        prefix = path.stat().st_size if path.exists() else 0
        assert prefix <= total
        size = 32 * 1024**2
        state = {'url': url, 'total': total, 'prefix': prefix,
                 'ranges': [[start, min(total, start + size) - 1]
                            for start in range(prefix, total, size)], 'done': []}
        manifest.write_text(json.dumps(state))
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)

    def fetch(pair):
        start, end = pair
        expected = f'bytes {start}-{end}/{total}'
        for attempt in range(6):
            try:
                request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    assert response.status == 206, response.status
                    assert response.headers['Content-Range'] == expected, response.headers
                    offset = start
                    while block := response.read(1024**2):
                        assert offset + len(block) <= end + 1
                        written = 0
                        while written < len(block):
                            written += os.pwrite(fd, block[written:], offset + written)
                        offset += len(block)
                    assert offset == end + 1, (offset, end)
                return start
            except Exception as exc:
                print(name, 'retry', start, attempt, repr(exc), flush=True)
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    try:
        pending = [pair for pair in state['ranges'] if pair[0] not in state['done']]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for start in pool.map(fetch, pending):
                state['done'].append(start)
                temp = manifest.with_suffix('.tmp')
                temp.write_text(json.dumps(state))
                temp.replace(manifest)
                print(name, f"{len(state['done'])}/{len(state['ranges'])} chunks", flush=True)
        os.fsync(fd)
        assert path.stat().st_size == total
    finally:
        os.close(fd)
    path.with_suffix(path.suffix + '.complete').touch()
    print(name, 'COMPLETE', total, flush=True)


if __name__ == '__main__':
    for domain in ['Movies_and_TV', 'Books', 'CDs_and_Vinyl']:
        download(f'reviews_{domain}_5.json.gz')
