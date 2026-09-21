"""Plain score distillation under a kernel-enforced filesystem read boundary."""
import ctypes
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from batching import loader as fast_loader

ROOT = Path(os.environ.get('CDR_RUN_ROOT', Path(__file__).resolve().parents[1]))
OUTPUT = ROOT / 'checkpoints/bb_learner'


def restrict_filesystem():
    # CUDA driver initialization opens host-mounted libraries outside standard library paths.
    # Initialize the device before restricting files; no model/data are loaded at this point.
    torch.cuda.init()
    libc = ctypes.CDLL(None, use_errno=True)
    probes = [ROOT / 'checkpoints/victim_source.pt', ROOT / 'checkpoints/base_seed2020.pt',
              ROOT / 'docs/idea.md']
    probes += list((ROOT / 'data/ready').glob('*/tgt_*/train_src.csv'))
    probes = [path for path in probes if path.exists()]
    assert probes

    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]

    handled = (1 << 13) - 1  # Landlock ABI v1 filesystem operations.
    readonly = (1 << 0) | (1 << 2) | (1 << 3)
    ruleset = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'Landlock ruleset unavailable; refuse unisolated extraction')
    allow = [(p, readonly) for p in ['/root/miniconda3', '/usr', '/lib', '/lib64', '/etc', '/sys', '/proc',
                                    str(ROOT / 'queries'), str(ROOT / 'configs'), str(Path(__file__).resolve().parent)]]
    allow += [('/dev', handled), ('/tmp', handled), (str(OUTPUT), handled)]
    for path, access in allow:
        if not Path(path).exists():
            continue
        parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
        rule = PathRule(access, parent)
        result = libc.syscall(445, fd, 1, ctypes.byref(rule), 0)
        os.close(parent)
        if result:
            raise OSError(ctypes.get_errno(), f'Landlock add path {path}')
    if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, fd, 0):
        raise OSError(ctypes.get_errno(), 'Landlock activation failed')
    os.close(fd)
    for path in probes:
        try:
            with open(path, 'rb'):
                pass
        except PermissionError:
            print('ACCESS_DENIED_AS_EXPECTED', str(path), flush=True)
        else:
            raise RuntimeError(f'Black-box isolation failed: {path}')


def main():
    OUTPUT.mkdir(exist_ok=True)
    restrict_filesystem()
    cfg = json.loads((ROOT / 'configs/blackbox.json').read_text())
    meta = json.loads((ROOT / 'queries/metadata.json').read_text())
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    response = np.load(ROOT / 'queries/topk_scores.npz')
    user_ids = np.repeat(response['user_ids'], cfg['k'])
    item_ids = response['item_ids'].reshape(-1)
    y = torch.tensor(response['scores'].reshape(-1), device='cuda')
    x = torch.tensor(np.column_stack([user_ids, item_ids]), dtype=torch.long, device='cuda')
    assert len(y) == meta['returned_score_count']
    loader = fast_loader((x, y), cfg['batch_size'])
    model = torch.nn.ModuleDict({
        'uid_embedding': torch.nn.Embedding(meta['num_source_users'], cfg['embedding_dim']),
        'iid_embedding': torch.nn.Embedding(meta['num_source_items'], cfg['embedding_dim'])}).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    history = []
    for epoch in range(cfg['epochs']):
        start = time.monotonic()
        total_loss = 0.
        for batch, target in loader:
            prediction = (model['uid_embedding'](batch[:, 0]) * model['iid_embedding'](batch[:, 1])).sum(1)
            loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(batch)
        row = {'epoch': epoch + 1, 'online_training_mse': float(total_loss / len(x)),
               'seconds': time.monotonic() - start}
        history.append(row)
        (OUTPUT / 'history.json').write_text(json.dumps(history, indent=2))
        torch.save(model.state_dict(), OUTPUT / 'surrogate_source.pt')
        print('EXTRACTION_EPOCH', json.dumps(row), flush=True)
    (OUTPUT / 'complete.json').write_text(json.dumps({'epochs': cfg['epochs'], 'isolation': 'Landlock ABI 1',
                                                     'method': cfg['method']}, indent=2))


if __name__ == '__main__':
    main()
