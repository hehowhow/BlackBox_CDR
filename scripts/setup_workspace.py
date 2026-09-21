"""Prepare runtime directories and pinned upstreams without copying experiment artifacts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh-workspace', type=Path,
                        help='Create a NEW code/config/docs-only directory; refuse existing paths')
    parser.add_argument('--with-dbgrme', action='store_true', help='Optional interface-reference repository')
    parser.add_argument('--skip-upstream', action='store_true', help='Prepare directories only; no network')
    args = parser.parse_args()
    root = BASE
    if args.fresh_workspace:
        root = args.fresh_workspace.resolve()
        root.mkdir(parents=True, exist_ok=False)
        for folder in ['scripts', 'configs', 'docs']:
            shutil.copytree(BASE/folder, root/folder,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ['README.md', '.gitignore', 'requirements.txt', 'upstream_repositories.json']:
            shutil.copy2(BASE/name, root/name)
    for folder in ['data/raw', 'data/mid', 'data/ready', 'data/id_maps', 'checkpoints',
                   'queries', 'results', 'logs', 'external', 'runs']:
        (root/folder).mkdir(parents=True, exist_ok=True)
    upstreams = json.loads((root/'upstream_repositories.json').read_text())
    if not args.skip_upstream:
        for name, spec in upstreams.items():
            if not spec['required'] and not args.with_dbgrme:
                continue
            path = root/'external'/name
            if not path.exists():
                subprocess.run(['git', 'clone', spec['url'], str(path)], check=True)
            current = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
            if current != spec['commit']:
                dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
                if dirty:
                    raise RuntimeError(f'Refuse to change a modified upstream checkout: {path}')
                subprocess.run(['git', '-C', str(path), 'checkout', '--detach', spec['commit']], check=True)
            print(name, spec['commit'])
    print('Workspace ready:', root)


if __name__ == '__main__':
    main()
