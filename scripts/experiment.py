"""Instrument official PFCDR; retain training/inversion objectives and MF voting."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(os.environ.get('CDR_RUN_ROOT', Path(__file__).resolve().parents[1]))
UPSTREAM = Path(__file__).resolve().parents[1] / 'external/PFCDR'
os.chdir(UPSTREAM)
sys.path.insert(0, str(UPSTREAM))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from run import Run
from models import FloatLookupEmbedding
from inversion import Inversion
from batching import loader as fast_loader

ORIGINAL_READ = Run.read_log_data


def read_batched(self, path, batchsize):
    original = ORIGINAL_READ(self, path, batchsize)
    return fast_loader(original.dataset.tensors, batchsize)


Run.read_log_data = read_batched


def seed(value=None):
    if value is None:
        value = config()['seed']
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def config():
    cfg = json.loads((UPSTREAM / 'config.json').read_text())
    cfg.update(json.loads((ROOT / 'configs/first_round.json').read_text()))
    cfg['root'] = str(ROOT / 'data') + '/'
    cfg['model_root'] = str(ROOT / 'checkpoints') + '/'
    return cfg


ORIGINAL_FORWARD = FloatLookupEmbedding.forward


def pooled_forward(self, x, stage=None):
    # Algebraically identical to sum(item_weight * x, dim=1), without B*I*D allocation.
    if x.dtype != torch.float32:
        return ORIGINAL_FORWARD(self, x, stage)
    uid = x[:, 0].long()
    pooled = x[:, 1:, 0] @ self.iid_embedding.weight
    if stage == 'save_inversed_iid':
        return torch.cat([uid.float(), pooled], dim=1).unsqueeze(1)
    return torch.cat([self.uid_embedding(uid), pooled.unsqueeze(1)], dim=1)


def verify_pooling():
    seed()
    layer = FloatLookupEmbedding(7, 31, 10).cuda()
    x = torch.cat([torch.arange(7, device='cuda').view(7, 1, 1),
                   torch.randn(7, 31, 1, device='cuda')], dim=1).requires_grad_()
    reference = ORIGINAL_FORWARD(layer, x)
    grad_ref = torch.autograd.grad(reference.square().sum(), x)[0]
    actual = pooled_forward(layer, x)
    grad_actual = torch.autograd.grad(actual.square().sum(), x)[0]
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(grad_actual, grad_ref, rtol=2e-5, atol=2e-5)
    print('PASS: pooled inversion forward and input gradient agree with official code', flush=True)


def checkpoint_path():
    return ROOT / 'checkpoints' / f"base_seed{config()['seed']}.pt"


def load_base(runner, source='wb'):
    model = runner.get_model()
    if source == 'wb':
        model.load_state_dict(torch.load(checkpoint_path(), map_location='cuda', weights_only=True))
    else:
        model.tgt_model.load_state_dict(torch.load(ROOT / 'checkpoints/target.pt', map_location='cuda', weights_only=True))
        model.rp_mapping.load_state_dict(torch.load(ROOT / 'checkpoints/mapping_init.pt', map_location='cuda', weights_only=True))
    return model


def train():
    while not (ROOT / 'data/split_manifest.json').exists():
        print('Waiting for data split', flush=True)
        time.sleep(20)
    seed()
    runner = Run(config())
    model = runner.get_model()
    source, target, test, _, _, _ = runner.get_data(runner.co_user_num)
    opt_source, opt_target, _ = runner.get_optimizer(model)
    criterion = torch.nn.MSELoss()
    records = []
    for domain, loader, optimizer, stage in [('tgt', target, opt_target, 'train_tgt'),
                                             ('src', source, opt_source, 'train_src')]:
        if runner.config.get('independent_domain_rng'):
            seed(runner.config['seed'] + (1 if domain == 'src' else 2))
        for epoch in range(runner.epoch):
            start = time.monotonic()
            runner.train(loader, model, criterion, optimizer, epoch, stage)
            record = {'domain': domain, 'epoch': epoch + 1, 'seconds': time.monotonic() - start}
            if domain == 'tgt':
                record['mae'], record['rmse'] = runner.eval_mae(model, test, 'test_tgt')
            records.append(record)
            save_json(ROOT / 'results/base_training.json', records)
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'domain': domain, 'epoch': epoch + 1, 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all()}, ROOT / 'checkpoints/base_progress.pt')
            print('EPOCH_RESULT', json.dumps(record), flush=True)
        if domain == 'tgt':
            inputs, labels = test.dataset.tensors
            with torch.no_grad():
                predictions = torch.cat([model(batch, 'test_tgt') for batch in inputs.split(4096)])
            np.savez_compressed(ROOT / 'results/tgt_predictions.npz', uid=inputs[:, 0].cpu().numpy(),
                                iid=inputs[:, 1].cpu().numpy(), y=labels[:, 0].cpu().numpy(),
                                prediction=predictions.cpu().numpy())
    torch.save(model.state_dict(), checkpoint_path())
    torch.save(model.tgt_model.state_dict(), ROOT / 'checkpoints/target.pt')
    torch.save(model.rp_mapping.state_dict(), ROOT / 'checkpoints/mapping_init.pt')
    # Service-side artifact: never passed to the BB learner.
    torch.save(model.src_model.state_dict(), ROOT / 'checkpoints/victim_source.pt')
    print('BASE_TRAINING_COMPLETE', flush=True)


def train_source_only():
    """Fresh source service for a direct-list run; no target MF is trained."""
    seed()
    runner = Run(config())
    model = runner.get_model()
    source = runner.read_log_data(runner.src_path, runner.batchsize_src)
    optimizer = torch.optim.Adam(model.src_model.parameters(), lr=runner.lr, weight_decay=runner.wd)
    if runner.config.get('independent_domain_rng'):
        seed(runner.config['seed'] + 1)
    records = []
    for epoch in range(runner.epoch):
        start = time.monotonic()
        runner.train(source, model, torch.nn.MSELoss(), optimizer, epoch, 'train_src')
        row = dict(domain='src', epoch=epoch + 1, seconds=time.monotonic() - start)
        records.append(row)
        save_json(ROOT / 'results/source_training.json', records)
        print('EPOCH_RESULT', json.dumps(row), flush=True)
    torch.save(model.src_model.state_dict(), ROOT / 'checkpoints/victim_source.pt')
    print('SOURCE_TRAINING_COMPLETE', flush=True)


def inverse(source):
    seed()
    verify_pooling()
    seed()
    FloatLookupEmbedding.forward = pooled_forward
    runner = Run(config())
    model = load_base(runner, source)
    if source == 'bb':
        model.src_model.load_state_dict(torch.load(ROOT / 'checkpoints/surrogate_source.pt',
                                                  map_location='cuda', weights_only=True))
    _, test_ids, train_ids = runner.get_data(runner.co_user_num, inversion=True)
    inv_cfg = json.loads((UPSTREAM / 'inversion_config.json').read_text())
    inv_cfg.update({'task': '1', 'src': runner.src, 'tgt': runner.tgt, 'ratio': runner.ratio,
                    'base_model': 'MF', 'src_item_dims': runner.iid_src, 'tgt_item_dims': runner.iid_tgt,
                    'syn_root': str(ROOT / 'checkpoints' / f'prototypes_{source}') + '/'})
    jobs = [('src', 'train', train_ids), ('src', 'test', test_ids)]
    if source == 'wb':
        jobs.append(('tgt', 'train', train_ids))
    for goal, stage, ids in jobs:
        output = prototype_prefix(source).parent / f'5_MF_{stage}_{goal}.csv'
        done = output.with_suffix('.done')
        if done.exists():
            print('Already complete:', output, flush=True)
            continue
        if output.exists():
            output.rename(output.with_suffix(f'.partial-{time.time_ns()}'))
        start = time.monotonic()
        Inversion(inv_cfg, model, ids, torch.device('cuda'), stage, goal).main()
        matrix = np.loadtxt(output, delimiter=',')
        assert matrix.shape == (len(ids), 11), matrix.shape
        assert np.isfinite(matrix).all()
        assert set(matrix[:, 0].astype(int)) == set(ids)
        save_json(done, {'users': len(ids), 'seconds': time.monotonic() - start})
        print('INVERSION_COMPLETE', goal, stage, len(ids), flush=True)


def prototype_prefix(source):
    cfg = config()
    pair = cfg['src_tgt_pairs'][cfg['task']]
    return ROOT / 'checkpoints' / f'prototypes_{source}' / f"8_2/tgt_{pair['tgt']}_src_{pair['src']}/5_MF"


@torch.no_grad()
def voting_predictions(model, prototypes, users, items, batch_size=64):
    # MF vote mean(dot(user_vectors, item)) == dot(mean(user_vectors), item).
    # Compute each cold user's vote once, rather than repeat for every rated item.
    mapped = model.rp_mapping(prototypes)
    all_users = model.tgt_model.uid_embedding.weight
    voted = []
    for batch in mapped.split(batch_size):
        distances = (all_users @ batch.T - 5).abs()
        indices = torch.sort(distances, dim=0).indices[:70000]
        voted.append(all_users[indices.T].mean(1))
    mean_users = torch.cat(voted)
    item_vectors = model.tgt_model.iid_embedding(items)
    return (mean_users[users] * item_vectors).sum(1)


def bridge(source):
    seed()
    runner = Run(config())
    model = load_base(runner, source)
    # Both branches start from exactly the same mapping and frozen target model.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.rp_mapping.parameters():
        parameter.requires_grad_(True)
    prefix = str(prototype_prefix(source))
    src_train = pd.read_csv(prefix + '_train_src.csv', header=None).to_numpy()
    tgt_train = pd.read_csv(str(prototype_prefix('wb')) + '_train_tgt.csv', header=None).to_numpy()
    assert np.array_equal(src_train[:, 0], tgt_train[:, 0])
    x = torch.tensor(np.column_stack([src_train, tgt_train[:, 1:]]), dtype=torch.float32, device='cuda')
    loader = DataLoader(TensorDataset(x, torch.arange(len(x), device='cuda')), batch_size=512, shuffle=True)
    test_proto = pd.read_csv(prefix + '_test_src.csv', header=None).to_numpy()
    ids = test_proto[:, 0].astype(int)
    id_lookup = {uid: i for i, uid in enumerate(ids)}
    test = pd.read_csv(runner.test_path, header=None).to_numpy()
    users = torch.tensor([id_lookup[int(u)] for u in test[:, 0]], device='cuda')
    items = torch.tensor(test[:, 1], dtype=torch.long, device='cuda')
    labels = torch.tensor(test[:, 2], dtype=torch.float32, device='cuda')
    prototypes = torch.tensor(test_proto[:, 1:], dtype=torch.float32, device='cuda')
    optimizer = torch.optim.Adam(model.rp_mapping.parameters(), lr=runner.lr_prototype, weight_decay=runner.wd)
    records = []
    for epoch in range(runner.epoch):
        runner.train(loader, model, torch.nn.MSELoss(), optimizer, epoch, 'train_source_free', mapping=True)
        model.eval()
        predictions = voting_predictions(model, prototypes, users, items)
        if epoch == 0:
            official_input = torch.cat([items[:3, None].float(), prototypes[users[:3]]], dim=1)
            with torch.no_grad():
                official = model(official_input, 'test_source_free')
            torch.testing.assert_close(predictions[:3], official, rtol=2e-5, atol=2e-5)
            print('PASS: cached MF voting agrees with official prediction', flush=True)
        record = {'epoch': epoch + 1, 'mae': (predictions - labels).abs().mean().item(),
                  'rmse': (predictions - labels).square().mean().sqrt().item()}
        records.append(record)
        save_json(ROOT / 'results' / f'{source}_training.json', records)
        print('BRIDGE_RESULT', source, json.dumps(record), flush=True)
    torch.save(model.rp_mapping.state_dict(), ROOT / 'checkpoints' / f'bridge_{source}.pt')
    np.savez_compressed(ROOT / 'results' / f'{source}_predictions.npz', uid=test[:, 0], iid=test[:, 1],
                        y=test[:, 2], prediction=predictions.cpu().numpy())
    print('BRIDGE_COMPLETE', source, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['verify', 'train', 'train_source', 'invert', 'bridge'])
    parser.add_argument('--source', choices=['wb', 'bb'], default='wb')
    args = parser.parse_args()
    if args.stage == 'verify':
        verify_pooling()
    elif args.stage == 'train':
        train()
    elif args.stage == 'train_source':
        train_source_only()
    elif args.stage == 'invert':
        inverse(args.source)
    elif args.stage == 'bridge':
        bridge(args.source)
