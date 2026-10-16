"""Check the staged source/result snapshot before publishing; no training needed."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]


def main():
    raw = subprocess.check_output(['git', '-C', str(BASE), 'ls-files', '-z'])
    paths = [Path(p.decode()) for p in raw.split(b'\0') if p]
    forbidden = {'data', 'checkpoints', 'queries', 'logs', 'external', '__pycache__', '.venv', '.git'}
    suffixes = {'.pt', '.pth', '.ckpt', '.npz', '.npy', '.gz', '.zip', '.tar', '.bin', '.pyc', '.log'}
    token_patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{30,}',
                      r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    total = 0
    largest = (0, '')
    for rel in paths:
        assert not (set(rel.parts) & forbidden), f'Forbidden tracked directory: {rel}'
        assert rel.suffix not in suffixes and not rel.name.startswith('.env'), rel
        assert not any(p.startswith('replay_') for p in rel.parts[:-1]), rel
        path = BASE/rel
        assert not path.is_symlink(), f'Unexpected tracked symlink: {rel}'
        size = path.stat().st_size
        is_figure = rel.suffix == '.png' and rel.parts[:2] == ('docs', 'figures')
        is_report_pdf = rel.suffix == '.pdf' and rel.parts[:1] == ('docs',)
        limit = 5 * 1024**2 if is_figure else 1024**2
        assert size < limit, f'Unexpected large file: {rel}'
        total += size
        largest = max(largest, (size, str(rel)))
        if is_figure:
            assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'), f'Invalid PNG: {rel}'
            continue
        if is_report_pdf:
            data = path.read_bytes()
            assert data.startswith(b'%PDF-') and b'%%EOF' in data[-1024:], f'Invalid PDF: {rel}'
            continue
        text = path.read_text()
        for pattern in token_patterns:
            assert not re.search(pattern, text), f'Possible credential in {rel}'
        if rel.suffix == '.py':
            ast.parse(text, filename=str(rel))
        if rel.suffix == '.json':
            json.loads(text)
    upstream = json.loads((BASE/'upstream_repositories.json').read_text())
    for entry in upstream.values():
        assert re.fullmatch('[0-9a-f]{40}', entry['commit'])
    for rel in ['README.md', 'docs/experiment_progress.md', 'docs/running.md',
                'docs/lapt.md', 'docs/lapt_paper_outline.md', 'docs/lapt_mf_full_seed2020.md']:
        for target in re.findall(r'\]\(([^)]+)\)', (BASE/rel).read_text()):
            if target.startswith(('https://', 'http://', '#')):
                continue
            resolved = ((BASE/rel).parent/target.split('#')[0]).resolve()
            assert resolved.exists(), f'Broken link in {rel}: {target}'
    with tempfile.TemporaryDirectory(prefix='cdr-publish-check-') as temp:
        fresh = Path(temp)/'fresh'
        subprocess.run([sys.executable, str(BASE/'scripts/setup_workspace.py'),
                        '--fresh-workspace', str(fresh), '--skip-upstream'], check=True)
        assert not list((fresh/'results').iterdir())
        assert not list((fresh/'checkpoints').iterdir())
        assert not list((fresh/'runs').iterdir())
        assert (fresh/'scripts/run_list_encoder.py').exists()
        assert (fresh/'data/raw').is_dir()
    print(json.dumps(dict(files=len(paths), total_bytes=total, largest_file=largest[1],
                          largest_bytes=largest[0], checks='syntax, JSON, artifacts, token patterns, links, fresh setup: PASS'), indent=2))


if __name__ == '__main__':
    main()
