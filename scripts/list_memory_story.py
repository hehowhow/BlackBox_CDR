"""Engram-inspired unordered-pair memory, added to the frozen global LIST protocol."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from small_task_comparison import PRIOR, TASK, SEEDS, BETAS, ResidualList, context, frame, corrections
from book_movie_protocol import BASE, digest, save, predict
from residual_experiment import seed_all
from list_source_backbones import student_guard, check_code, read
from residual_summary import metrics

PARENT = BASE/'runs/small_task_comparison_8650'
VARIANTS = {'pair20': 20, 'pair50': 50}


def parameter_hash(model, shared=False):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not shared or not name.startswith('pair_memory.'):
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def pair_keys(ids, depth, vocabulary):
    """Rank is not temporal order. The pair key is invariant to list order."""
    a, b = np.triu_indices(depth, 1)
    left, right = ids[:, a], ids[:, b]
    return np.minimum(left, right).astype(np.int64)*vocabulary + np.maximum(left, right)


def build_memory(ids, train_users, depth, vocabulary, minimum, capacity):
    keys = pair_keys(ids, depth, vocabulary)
    # Top-K contains unique items; each pair occurs at most once per user.
    eligible, support = np.unique(keys[train_users], return_counts=True)
    mask = support >= minimum
    eligible, support = eligible[mask], support[mask]
    order = np.lexsort((eligible, -support))[:capacity]
    selected, counts = eligible[order], support[order]
    order = np.argsort(selected)
    selected, counts = selected[order], counts[order]
    rows = np.zeros(keys.shape, dtype=np.int64)
    if len(selected):
        positions = np.searchsorted(selected, keys)
        valid = (positions < len(selected)) & (selected[np.minimum(positions, len(selected)-1)] == keys)
        rows[valid] = positions[valid]+1
    # Compact each user's matches, retaining deterministic pair enumeration order.
    hits = (rows != 0).sum(1)
    compact = np.zeros((len(ids), max(1, int(hits.max()))), dtype=np.int64)
    rid, cid = np.nonzero(rows)
    destinations = (rows != 0).cumsum(1)[rid, cid]-1
    compact[rid, destinations] = rows[rid, cid]
    return selected, counts, compact, hits


class MemoryResidual(ResidualList):
    def __init__(self, lists, vocab, items, base, cfg, pair_rows):
        super().__init__(lists, vocab, items, base, cfg)
        self.pair_memory = nn.EmbeddingBag(cfg['memory_capacity']+1, cfg['dim'], mode='sum', padding_idx=0)
        nn.init.normal_(self.pair_memory.weight, std=.1)
        with torch.no_grad():
            self.pair_memory.weight[0].zero_()
        self.register_buffer('pair_rows', pair_rows, persistent=False)
        self.register_buffer('pair_counts', (pair_rows != 0).sum(1).clamp(min=1), persistent=False)

    def memory_pool(self, users):
        valid = users < len(self.pair_rows)
        result = self.source_items.weight.new_zeros((len(users), self.source_items.embedding_dim))
        if valid.any():
            indices = valid.nonzero(as_tuple=True)[0]
            u = users[indices]
            values = self.pair_memory(self.pair_rows[u])/self.pair_counts[u, None]
            result = result.index_copy(0, indices, values)
        return result

    def correction(self, u, i):
        rows = self.lists[u]
        h = self.source_items(rows, per_sample_weights=self.rank_weights.expand_as(rows))
        z = self.encoder(h+self.memory_pool(u))*(rows[:, :1] != 0)
        return (z*self.target_items(i)).sum(-1)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    prior = read(PARENT/'protocol.json')
    assert read(PARENT/'verification.json')['status'] == read(PARENT/'summary.json')['status'] == 'PASS'
    for path, expected in prior['code_hashes'].items():
        assert digest(Path(path)) == expected, path
    paths = [Path(__file__), BASE/'scripts/slurm_list_memory_story.sh']
    cfg = dict(prior, variants=VARIANTS, arms=['real'], memory_capacity=8192, memory_minimum_users=3,
        parent=str(PARENT), parent_protocol_sha256=digest(PARENT/'protocol.json'),
        architecture='global rank-weighted Top100 pool h plus uniform mean g of matching pair embeddings; original MLP(h+g); original item-mean residual head',
        memory_dictionary='unordered pairs from first20 or first50; >=3 distinct phase-training users; top8192 by support descending then pair key ascending; unknown=zero',
        memory_training='select dictionary uses inner_train users; refit dictionary rebuilt using outer_train users; source lists of validation/test users never contribute counts',
        initialization='all original tensors identical; fixed8193x32 memory initialized normal std.1, padding0 zero; zero original final encoder layer; all named parameter tensors identical before select/refit',
        parameter_change=8193*prior['dim'],
        scope='Engram-inspired simple associative memory, not a reproduction of full Engram; no same-capacity MLP control; historical test exploratory',
        code_hashes={**prior['code_hashes'], **{str(p.resolve()): digest(p) for p in paths}})
    save(root/'protocol.json', cfg)
    for variant in VARIANTS:
        vroot = root/variant
        vroot.mkdir()
        save(vroot/'protocol.json', dict(cfg, variant=variant))
        for beta in BETAS:
            cell, old = vroot/f'beta_{beta}', PARENT/f'beta_{beta}'
            cell.mkdir()
            for d in ['data', 'baseline', 'inner_baseline']:
                (cell/d).symlink_to((old/d).resolve(), target_is_directory=True)
            manifest = read(old/'manifest.json')
            for path, expected in manifest['hashes'].items():
                assert digest(old/path) == expected
            save(cell/'manifest.json', manifest)
            for seed in SEEDS:
                (cell/f'seed_{seed}/residual_real').mkdir(parents=True)
    save(root/'data_verification.json', dict(status='PASS', same_splits_and_baselines=True, parent=8650))
    print('MEMORY_PREPARED', root, flush=True)


def selftest(root):
    check_code(root)
    cfg = read(root/'protocol.json')
    # A pair supported by held-out users alone cannot enter the training dictionary.
    ids = np.array([[1,2,3], [1,2,4], [1,8,9], [1,8,9]])
    keys, counts, rows, hits = build_memory(ids, np.array([0,1]), 3, 10, 2, 8)
    np.testing.assert_array_equal(keys, [12])
    np.testing.assert_array_equal(counts, [2])
    np.testing.assert_array_equal(hits, [1,1,0,0])
    np.testing.assert_array_equal(np.sort(pair_keys(ids,3,10)), np.sort(pair_keys(ids[:,::-1],3,10)))
    lists = torch.tensor([[1,2,3], [2,1,4], [0,0,0]], device='cuda')
    pair_rows = torch.tensor([[1,2], [1,0]], device='cuda')
    base = torch.tensor([3.,4.,2.], device='cuda')
    users = torch.arange(3, device='cuda')
    seed_all(123)
    original = ResidualList(lists, 4, 3, base, cfg).cuda()
    seed_all(123)
    model = MemoryResidual(lists, 4, 3, base, cfg, pair_rows).cuda()
    assert parameter_hash(model, True) == parameter_hash(original)
    model.eval()
    torch.testing.assert_close(model(users, users), base, rtol=0, atol=0)
    manual = torch.stack([model.pair_memory.weight[[1,2]].mean(0), model.pair_memory.weight[1], torch.zeros(cfg['dim'], device='cuda')])
    torch.testing.assert_close(model.memory_pool(users), manual)
    with torch.no_grad():
        model.encoder[-1].weight.normal_(std=.1)
        model.encoder[-1].bias.fill_(1.)
    (model(users, users)-5).square().mean().backward()
    assert model.pair_memory.weight.grad[1:].abs().sum() > 0
    assert model.pair_memory.weight.grad[0].abs().sum() == 0
    assert model(users, users)[2] == base[2]
    save(root/'selftest.json', dict(status='PASS', training_only_dictionary=True, unordered_pair_invariance=True,
        manual_pool_match=True, memory_receives_gradient=True, padding_zero_gradient=True,
        shared_initialization_matches_parent=True, zero_initial_correction=True, no_list_fallback=True))
    print('MEMORY_SELFTEST_PASS', flush=True)


def train(root, variant, beta, seed):
    vroot = root/variant
    check_code(vroot)
    student_guard(vroot)
    cell, cfg, manifest = context(vroot, beta)
    out = cell/f'seed_{seed}/residual_real'
    assert not (out/'started.json').exists()
    save(out/'started.json', dict(job=os.environ['SLURM_JOB_ID'], gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    query = PRIOR/TASK/f'seed_{seed}/queries/DNN'
    assert digest(query/'topk_ids.npz') == read(query/'complete.json')['item_ids_sha256']
    api = np.load(query/'topk_ids.npz')
    np.testing.assert_array_equal(api['user_ids'], np.arange(manifest['overlap_users']))
    ids = api['item_ids']
    assert ids.shape[1] == 100
    assert np.all(np.diff(np.sort(ids, axis=1), axis=1) != 0)
    vocabulary = np.unique(ids)
    compact = np.searchsorted(vocabulary, ids)+1
    lists = torch.zeros((manifest['target_users'], cfg['k']), device='cuda', dtype=torch.long)
    lists[:len(ids)] = torch.tensor(compact, device='cuda')
    val = frame(cell/'data/validation.csv')
    vx = torch.tensor(val[:,:2], device='cuda', dtype=torch.long)
    tx = torch.tensor(np.load(cell/'data/test_features.npy'), device='cuda', dtype=torch.long)
    initial_hashes, memory_info = [], {}
    selection = None
    for phase in ['select','refit']:
        name = 'inner_train' if phase == 'select' else 'train'
        data = frame(cell/f'data/{name}.csv')
        train_users = np.load(cell/f'data/{name}_users.npy')
        actual_users = np.unique(data[:,0].astype(int))
        np.testing.assert_array_equal(np.sort(train_users), actual_users[actual_users < len(ids)])
        assert not np.intersect1d(train_users, np.load(cell/'data/test_users.npy')).size
        if phase == 'select':
            assert not np.intersect1d(train_users, np.load(cell/'data/validation_users.npy')).size
        tick = time.monotonic()
        keys, support, rows, hits = build_memory(compact, train_users, VARIANTS[variant], len(vocabulary)+1,
            cfg['memory_minimum_users'], cfg['memory_capacity'])
        np.savez_compressed(out/f'{phase}_memory.npz', keys=keys, support=support, training_users=train_users,
            pair_rows=rows, vocabulary=vocabulary)
        groups = {g: np.load(cell/f'data/{g}_users.npy') for g in ['inner_train','validation','test']}
        coverage = {g:dict(user_hit_fraction=float((hits[u]>0).mean()), mean_matched_pairs=float(hits[u].mean())) for g,u in groups.items()}
        memory_info[phase] = dict(entries=len(keys), minimum_support=int(support.min()) if len(support) else 0,
            maximum_support=int(support.max()) if len(support) else 0, training_users=len(train_users),
            coverage=coverage, construction_seconds=time.monotonic()-tick, dictionary_sha256=digest(out/f'{phase}_memory.npz'))
        save(out/'memory_info.json', memory_info)
        pair_rows = torch.tensor(rows, device='cuda')
        basepath = cell/('inner_baseline' if phase == 'select' else 'baseline')
        stats = dict(np.load(basepath/'statistics.npz'))
        base64 = predict(stats, np.arange(manifest['target_items']), 'item_mean')
        base = torch.tensor(base64.astype(np.float32), device='cuda')
        seen = torch.tensor(np.load(basepath/'seen.npy'), device='cuda')
        seed_all(seed)
        model = MemoryResidual(lists, len(vocabulary), manifest['target_items'], base, cfg, pair_rows).cuda()
        initial_hashes.append(parameter_hash(model))
        shared_hash = parameter_hash(model, True)
        parent = read(PARENT/f'beta_{beta}/seed_{seed}/residual_real/complete.json')
        assert shared_hash == parent['initial_parameters_sha256']
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert initial_hashes[0] == initial_hashes[-1]
        seed_all(seed+100)
        x = torch.tensor(data[:,:2], device='cuda', dtype=torch.long)
        y = torch.tensor(data[:,2], device='cuda', dtype=torch.float32)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(seed+200)
        if phase == 'select':
            np.testing.assert_array_equal(corrections(model,vx,seen), np.zeros(len(vx), dtype=np.float32))
            baseval = base64[val[:,1].astype(int)]
            selection = dict(epoch=0, lambda_value=0., validation_rmse=float(np.sqrt(np.square(baseval-val[:,2]).mean())))
            torch.save(model.state_dict(), out/'selected.pt')
            np.save(out/'validation.npy', baseval)
        epochs = cfg['epochs'] if phase == 'select' else selection['epoch']
        history = []
        for epoch in range(1, epochs+1):
            tick = time.monotonic()
            model.train()
            total = torch.zeros((), device='cuda')
            for indices in torch.randperm(len(x), device='cuda', generator=generator).split(cfg['batch_size']):
                opt.zero_grad(set_to_none=True)
                loss = (model(x[indices,0], x[indices,1])-y[indices]).square().mean()
                loss.backward()
                opt.step()
                total += loss.detach()*len(indices)
            row = dict(epoch=epoch, train_mse=total.item()/len(x), seconds=time.monotonic()-tick)
            assert np.isfinite(row['train_mse'])
            if phase == 'select':
                correction = corrections(model,vx,seen)
                row['validation'] = {}
                for lam in cfg['lambdas']:
                    pred = baseval+lam*correction
                    rmse = float(np.sqrt(np.square(pred-val[:,2]).mean()))
                    row['validation'][str(lam)] = rmse
                    if rmse < selection['validation_rmse']:
                        selection = dict(epoch=epoch, lambda_value=lam, validation_rmse=rmse)
                        torch.save(model.state_dict(), out/'selected.pt')
                        np.save(out/'validation.npy', pred)
            history.append(row)
            save(out/f'{phase}_history.json', history)
            print('MEMORY',variant,beta,seed,phase,json.dumps(row),flush=True)
        if phase == 'select':
            model.load_state_dict(torch.load(out/'selected.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(baseval+selection['lambda_value']*corrections(model,vx,seen),np.load(out/'validation.npy'))
            save(out/'selection.json', selection)
        else:
            torch.save(model.state_dict(), out/'final.pt')
            pred = base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(model,tx,seen)
            np.save(out/'test.npy', pred)
            restored = MemoryResidual(lists,len(vocabulary),manifest['target_items'],base,cfg,pair_rows).cuda()
            restored.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(restored,tx,seen),pred)
            save(out/'complete.json',dict(status='PASS',variant=variant,beta=beta,seed=seed,
                selection_sha256=digest(out/'selection.json'), final_sha256=digest(out/'final.pt'), test_sha256=digest(out/'test.npy'),
                memory_sha256=digest(out/'memory_info.json'), initial_parameters_sha256=initial_hashes[0],
                shared_initial_parameters_sha256=shared_hash, trainable_parameters=trainable, additional_parameters=cfg['parameter_change'],
                full_outer_train_rows=len(data),refit_epochs=epochs,refit_reinitialized=True,restored_predictions=True,test_labels_opened=False))
            del restored
        del model, opt, x, y, data, pair_rows
    print('MEMORY_FROZEN',variant,beta,seed,flush=True)


def evaluate(root):
    check_code(root)
    cfg = read(root/'protocol.json')
    assert read(root/'selftest.json')['status'] == 'PASS'
    frozen = {}
    for variant in VARIANTS:
        for beta in BETAS:
            for seed in SEEDS:
                out = root/variant/f'beta_{beta}/seed_{seed}/residual_real'
                complete = read(out/'complete.json')
                assert complete['status'] == 'PASS' and not complete['test_labels_opened']
                for name,key in [('final.pt','final_sha256'),('test.npy','test_sha256'),('selection.json','selection_sha256'),('memory_info.json','memory_sha256')]:
                    assert digest(out/name) == complete[key]
                parent = read(PARENT/f'beta_{beta}/seed_{seed}/residual_real/complete.json')
                assert complete['shared_initial_parameters_sha256'] == parent['initial_parameters_sha256']
                for phase in ['select','refit']:
                    assert digest(out/f'{phase}_memory.npz') == read(out/'memory_info.json')[phase]['dictionary_sha256']
                frozen[str(out.relative_to(root))] = digest(out/'complete.json')
    with (root/'evaluation_started.json').open('x') as stream:
        json.dump(frozen,stream,indent=2)
    rows, seeds, results = [], [], {}
    for beta in BETAS:
        cell = PARENT/f'beta_{beta}'
        manifest = read(cell/'manifest.json')
        assert digest(cell/'data/test.csv') == manifest['hashes']['data/test.csv']
        test = frame(cell/'data/test.csv')
        np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
        y = test[:,2]
        _, users = np.unique(test[:,0],return_inverse=True)
        original = [metrics(np.load(cell/f'seed_{s}/residual_real/test.npy'),y,users) for s in SEEDS]
        original_mean = {k:float(np.mean([v[k] for v in original])) for k in ['mae','rmse']}
        mean = metrics(predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean'),y,users)
        for method,values in [('item_mean',[mean]*3),('original_LIST',original)]:
            r=dict(beta=beta,variant=method,mae=float(np.mean([v['mae'] for v in values])),rmse=float(np.mean([v['rmse'] for v in values])),
                mae_std=float(np.std([v['mae'] for v in values],ddof=1)),rmse_std=float(np.std([v['rmse'] for v in values],ddof=1)))
            r.update(delta_mae=r['mae']-original_mean['mae'],delta_rmse=r['rmse']-original_mean['rmse'])
            rows.append(r)
        results[str(beta)] = {}
        for variant in VARIANTS:
            values = []
            for index,seed in enumerate(SEEDS):
                out = root/variant/f'beta_{beta}/seed_{seed}/residual_real'
                value = metrics(np.load(out/'test.npy'),y,users)
                selected = read(out/'selection.json')
                r=dict(variant=variant,beta=beta,seed=seed,mae=value['mae'],rmse=value['rmse'],
                    delta_mae=value['mae']-original[index]['mae'],delta_rmse=value['rmse']-original[index]['rmse'],
                    epoch=selected['epoch'],lambda_value=selected['lambda_value'],validation_rmse=selected['validation_rmse'])
                seeds.append(r)
                values.append(value)
            r=dict(beta=beta,variant=variant,mae=float(np.mean([v['mae'] for v in values])),rmse=float(np.mean([v['rmse'] for v in values])),
                mae_std=float(np.std([v['mae'] for v in values],ddof=1)),rmse_std=float(np.std([v['rmse'] for v in values],ddof=1)))
            r.update(delta_mae=r['mae']-original_mean['mae'],delta_rmse=r['rmse']-original_mean['rmse'])
            rows.append(r)
            results[str(beta)][variant]=r
    for name,values in [('comparison',rows),('per_seed',seeds)]:
        with (root/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    save(root/'summary.json',dict(status='PASS',results=results,comparison=rows,protocol=cfg))
    save(root/'verification.json',dict(status='PASS',models=12,all_frozen_before_test=True,shared_initialization_matches_original=True,
        same_splits_and_baselines=True,training_only_pair_dictionary=True,restored_predictions=True,additional_parameters=cfg['parameter_change']))
    print('MEMORY_ALL_PASS',json.dumps(rows),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','selftest','train','evaluate'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--beta',type=int,choices=BETAS)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.stage == 'train':
        train(args.root.resolve(),args.variant,args.beta,args.seed)
    else:
        globals()[args.stage](args.root.resolve())
