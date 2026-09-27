"""Fixed response-depth LIST study; isolated teacher export and unchanged student.

This is a list-length acquisition experiment, not an active-query policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from small_task_comparison import (PRIOR, TASK, SEEDS, BETAS, ResidualList,
                                   context, frame, corrections)
from book_movie_protocol import BASE, digest, save, predict
from residual_experiment import seed_all, guard_test_labels
from list_source_backbones import student_guard, check_code, read
from residual_summary import metrics

PARENT = BASE/'runs/small_task_comparison_8650'
DEPTHS = [20, 50, 200]


def parameter_hash(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    cfg = read(PARENT/'protocol.json')
    assert read(PARENT/'verification.json')['status'] == 'PASS'
    for p, h in cfg['code_hashes'].items():
        assert digest(Path(p)) == h, p
    own = [Path(__file__), BASE/'scripts/slurm_list_acquisition_story.sh',
           BASE/'scripts/verify_list_acquisition_story.py']
    cfg = dict(cfg, variants=[f'k_{k}' for k in DEPTHS], depths=DEPTHS,
        arms=['real'], backbones=['DNN'],
        parent=str(PARENT), parent_protocol_sha256=digest(PARENT/'protocol.json'),
        study='fixed source response depth; no active query or stopping policy',
        source_policy='same frozen DNN checkpoint; separate teacher process exports Top200 IDs/ranks only',
        export_batch=128, export_k=200,
        vocabulary='per-seed union of legal Top200 returned IDs; original sorted Top100 IDs first, new IDs appended; shared by all depths',
        initialization='preserve every original Top100 source vector and every encoder/target tensor; independent RNG for appended source IDs',
        interface_extension='K200 requires source service to allow deeper output; K20/50 are cache-prefix simulations',
        student_access='own per-user K-list, common opaque ID vocabulary and random initialization only; no other per-user depth list, teacher weights or scores',
        budget='report logical per-user requests, total returned IDs, and full-catalog scored candidates; actual research export200 counted separately',
        frozen_design='K20/50/200 fixed before training; all variants frozen before test labels; no test-based variant selection',
        code_hashes={**cfg['code_hashes'], **{str(p.resolve()):digest(p) for p in own}})
    save(root/'protocol.json', cfg)
    save(root/'manifest.json', read(PRIOR/TASK/'manifest.json'))
    for seed in SEEDS:
        (root/f'queries/seed_{seed}').mkdir(parents=True)
    for k in DEPTHS:
        vr = root/f'k_{k}'; vr.mkdir()
        save(vr/'protocol.json', dict(cfg, k=k, variant=f'k_{k}'))
        for beta in BETAS:
            old = PARENT/f'beta_{beta}'; cell = vr/f'beta_{beta}'; cell.mkdir()
            for d in ['data', 'baseline', 'inner_baseline']:
                (cell/d).symlink_to((old/d).resolve(), target_is_directory=True)
            m = read(old/'manifest.json')
            for p, h in m['hashes'].items():
                assert digest(old/p) == h, p
            save(cell/'manifest.json', m)
            for seed in SEEDS:
                (cell/f'seed_{seed}/residual_real').mkdir(parents=True)
    save(root/'data_verification.json', dict(status='PASS', identical_parent_splits=True,
        same_ordinary_item_means=True, design_fixed_before_training=True))
    print('ACQUISITION_PREPARED', root, flush=True)


@torch.no_grad()
def export(root, seed):
    """Only this process is allowed to open the frozen source checkpoint."""
    from pfcdr_backbone_study import make_model, effective_users, table
    guard_test_labels(root); check_code(root)
    cfg = read(root/'protocol.json'); m = read(root/'manifest.json')
    old = PRIOR/TASK/f'seed_{seed}/queries/DNN'; record = read(old/'complete.json')
    assert digest(old/'source.pt') == record['source_sha256']
    assert digest(old/'topk_ids.npz') == record['item_ids_sha256']
    old_ids = np.load(old/'topk_ids.npz')['item_ids']
    seed_all(seed); model = make_model('DNN', m, cfg['source_dim'])
    model.src_model.load_state_dict(torch.load(old/'source.pt', map_location='cuda', weights_only=True))
    model.eval(); u = effective_users(model, 'DNN', 'src')[:m['overlap_users']]
    v = table(model, 'DNN', 'src').iid_embedding.weight
    rows = []; mismatches = 0; canonicalized = 0; max_prefix_score_difference = 0.
    tick = time.monotonic()
    for start in range(0, len(u), cfg['export_batch']):
        scores = u[start:start+cfg['export_batch']] @ v.T
        values, indices = scores.topk(200, dim=1, sorted=True)
        ids = indices.cpu().numpy().astype(np.int32)
        reference = old_ids[start:start+len(ids)]
        mismatches += int((ids[:, :100] != reference).sum())
        if not np.array_equal(ids[:, :100], reference):
            # A different top-k kernel may order numerical ties differently.
            # Preserve the published prefix only after checking rankwise scores.
            old_scores = scores.gather(1, torch.tensor(reference, device='cuda', dtype=torch.long))
            error = float((old_scores-values[:, :100]).abs().max())
            max_prefix_score_difference = max(max_prefix_score_difference, error)
            assert error <= 4e-6, ('non-tie prefix drift', seed, start, error)
            for j in range(len(ids)):
                if not np.array_equal(ids[j, :100], reference[j]):
                    assert np.isin(reference[j], ids[j]).all()
                    tail = ids[j][~np.isin(ids[j], reference[j])]
                    assert len(tail) == 100
                    ids[j] = np.concatenate([reference[j], tail]); canonicalized += 1
        np.testing.assert_array_equal(ids[:, :100], reference)
        ordered = scores.gather(1, torch.tensor(ids, device='cuda', dtype=torch.long))
        assert bool((ordered[:, 1:]-ordered[:, :-1] <= 4e-6).all())
        assert bool((ordered.min(1).values >= values[:, -1]-4e-6).all())
        rows.append(ids)
    ids = np.concatenate(rows); np.testing.assert_array_equal(ids[:, :100], old_ids)
    assert (np.diff(np.sort(ids, axis=1), axis=1) > 0).all()
    out = root/f'queries/seed_{seed}'
    hashes = {}
    for k in [20, 50, 100, 200]:
        path = out/f'k_{k}_ids.npz'
        np.savez_compressed(path, user_ids=np.arange(len(ids), dtype=np.int32), item_ids=ids[:, :k])
        hashes[str(k)] = digest(path)
    old_vocab = np.unique(old_ids); new_vocab = np.setdiff1d(np.unique(ids), old_vocab)
    np.save(out/'vocabulary.npy', np.concatenate([old_vocab, new_vocab]))
    save(out/'complete.json', dict(status='PASS', source_sha256=record['source_sha256'],
        original_query_sha256=record['item_ids_sha256'], ids_sha256=hashes,
        vocabulary_sha256=digest(out/'vocabulary.npy'), old_vocabulary_size=len(old_vocab),
        new_vocabulary_size=len(new_vocab), vocabulary_size=len(old_vocab)+len(new_vocab),
        exact_preserved_top100_prefix=True, raw_prefix_mismatched_positions=mismatches,
        numerically_tied_rows_canonicalized=canonicalized,
        maximum_prefix_score_difference=max_prefix_score_difference,
        no_scores_exported=True, seen_item_filter=False, model_frozen=True,
        users=len(ids), source_items=len(v), export_k=200,
        logical_requests=len(ids), returned_ids=int(ids.size), scored_candidates=int(len(ids)*len(v)),
        seconds=time.monotonic()-tick))
    print('TEACHER_EXPORT_PASS', seed, 'ties', canonicalized, flush=True)


def initialize(root, seed):
    """Public ID-only initialization process. It never opens source weights."""
    student_guard(root); check_code(root)
    cfg = read(root/'protocol.json'); m = read(root/'manifest.json')
    q = root/f'queries/seed_{seed}'; qr = read(q/'complete.json')
    assert digest(q/'vocabulary.npy') == qr['vocabulary_sha256']
    vocab = np.load(q/'vocabulary.npy'); original_n = qr['old_vocabulary_size']
    base = torch.zeros(m['target_items'], device='cuda')
    dummy = torch.zeros((1, 100), device='cuda', dtype=torch.long)
    seed_all(seed); original = ResidualList(dummy, original_n, m['target_items'], base, cfg).cuda()
    original_hash = parameter_hash(original)
    for beta in BETAS:
        assert original_hash == read(PARENT/f'beta_{beta}/seed_{seed}/residual_real/complete.json')['initial_parameters_sha256']
    seed_all(seed); expanded = ResidualList(dummy, len(vocab), m['target_items'], base, cfg).cuda()
    old = dict(original.named_parameters())
    with torch.no_grad():
        for name, p in expanded.named_parameters():
            if name == 'source_items.weight':
                p[:original_n+1].copy_(old[name])
                gen = torch.Generator(device='cuda').manual_seed(seed+99000)
                p[original_n+1:].normal_(std=.1, generator=gen)
                torch.testing.assert_close(p[:original_n+1], old[name], rtol=0, atol=0)
            else:
                p.copy_(old[name]); torch.testing.assert_close(p, old[name], rtol=0, atol=0)
    # Only parameter tensors are stored: rank weights depend on each variant's K.
    initial = {name:p.detach().cpu() for name,p in expanded.named_parameters()}
    torch.save(initial, q/'initial_student.pt')
    save(q/'initialization.json', dict(status='PASS', original_parameters_sha256=original_hash,
        initial_parameters_sha256=parameter_hash(expanded), initial_file_sha256=digest(q/'initial_student.pt'),
        old_source_and_all_target_encoder_tensors_exactly_preserved=True,
        vocabulary_size=len(vocab), trainable_parameters=sum(p.numel() for p in expanded.parameters() if p.requires_grad),
        added_parameters=(len(vocab)-original_n)*cfg['dim'], no_teacher_or_target_label_access=True))
    print('SHARED_INITIALIZATION_PASS', seed, flush=True)


def guard_other_lists(root, seed, k):
    allowed = (root/f'queries/seed_{seed}/k_{k}_ids.npz').resolve()
    def hook(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if (p.name.endswith('_ids.npz') or p.name == 'topk_ids.npz') and p != allowed:
                raise RuntimeError(f'Student may only read its own K-list: {p}')
    sys.addaudithook(hook)


def train(root, seed, beta, k):
    study = root; root = root/f'k_{k}'
    check_code(root); student_guard(root); guard_other_lists(study, seed, k)
    cell, cfg, m = context(root, beta); out = cell/f'seed_{seed}/residual_real'
    assert not (out/'started.json').exists()
    save(out/'started.json', dict(job=os.environ['SLURM_JOB_ID'], gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        device=torch.cuda.get_device_name(), k=k, epoch_budget=10))
    q = study/f'queries/seed_{seed}'; qr = read(q/'complete.json'); init = read(q/'initialization.json')
    assert digest(q/f'k_{k}_ids.npz') == qr['ids_sha256'][str(k)]
    assert digest(q/'vocabulary.npy') == qr['vocabulary_sha256']
    assert digest(q/'initial_student.pt') == init['initial_file_sha256']
    api = np.load(q/f'k_{k}_ids.npz'); ids = api['item_ids']; vocab = np.load(q/'vocabulary.npy')
    np.testing.assert_array_equal(api['user_ids'], np.arange(m['overlap_users']))
    assert ids.shape == (m['overlap_users'], k)
    mapping = np.zeros(int(vocab.max())+1, dtype=np.int64); mapping[vocab] = np.arange(1, len(vocab)+1)
    compact = mapping[ids]; assert (compact > 0).all()
    lists = torch.zeros((m['target_users'], k), device='cuda', dtype=torch.long)
    lists[:len(ids)] = torch.tensor(compact, device='cuda')
    val = frame(cell/'data/validation.csv'); vx = torch.tensor(val[:, :2], device='cuda', dtype=torch.long)
    tx = torch.tensor(np.load(cell/'data/test_features.npy'), device='cuda', dtype=torch.long)
    selection = None; init_hashes = []; order_hashes = {}
    for phase in ['select', 'refit']:
        data = frame(cell/('data/inner_train.csv' if phase == 'select' else 'data/train.csv'))
        assert not np.intersect1d(np.unique(data[:,0]), np.load(cell/'data/test_users.npy')).size
        if phase == 'select':
            assert not np.intersect1d(np.unique(data[:,0]), np.load(cell/'data/validation_users.npy')).size
        basepath = cell/('inner_baseline' if phase == 'select' else 'baseline')
        base64 = predict(dict(np.load(basepath/'statistics.npz')), np.arange(m['target_items']), 'item_mean')
        base = torch.tensor(base64.astype(np.float32), device='cuda')
        seen = torch.tensor(np.load(basepath/'seen.npy'), device='cuda')
        seed_all(seed); model = ResidualList(lists, len(vocab), m['target_items'], base, cfg).cuda()
        params = torch.load(q/'initial_student.pt', map_location='cuda', weights_only=True)
        with torch.no_grad():
            for name, p in model.named_parameters(): p.copy_(params[name])
        init_hashes.append(parameter_hash(model)); assert init_hashes[-1] == init['initial_parameters_sha256']
        seed_all(seed+100)
        x = torch.tensor(data[:, :2], device='cuda', dtype=torch.long)
        y = torch.tensor(data[:, 2], device='cuda', dtype=torch.float32)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        gen = torch.Generator(device='cuda').manual_seed(seed+200)
        order_hash = hashlib.sha256()
        if phase == 'select':
            np.testing.assert_array_equal(corrections(model, vx, seen), np.zeros(len(vx), dtype=np.float32))
            baseval = base64[val[:, 1].astype(int)]
            selection = dict(epoch=0, lambda_value=0., validation_rmse=float(np.sqrt(np.square(baseval-val[:, 2]).mean())))
            torch.save(model.state_dict(), out/'selected.pt'); np.save(out/'validation.npy', baseval)
        epochs = cfg['epochs'] if phase == 'select' else selection['epoch']; history = []
        for epoch in range(1, epochs+1):
            tick = time.monotonic(); model.train(); total = torch.zeros((), device='cuda')
            order = torch.randperm(len(x), device='cuda', generator=gen)
            order_hash.update(order.cpu().numpy().tobytes())
            for ix in order.split(cfg['batch_size']):
                opt.zero_grad(set_to_none=True)
                loss = (model(x[ix,0], x[ix,1])-y[ix]).square().mean()
                loss.backward(); opt.step(); total += loss.detach()*len(ix)
            row = dict(epoch=epoch, train_mse=total.item()/len(x), seconds=time.monotonic()-tick)
            assert np.isfinite(row['train_mse'])
            if phase == 'select':
                correction = corrections(model, vx, seen); row['validation'] = {}
                for lam in cfg['lambdas']:
                    pred = baseval+lam*correction; rmse = float(np.sqrt(np.square(pred-val[:,2]).mean()))
                    row['validation'][str(lam)] = rmse
                    if rmse < selection['validation_rmse']:
                        selection = dict(epoch=epoch, lambda_value=lam, validation_rmse=rmse)
                        torch.save(model.state_dict(), out/'selected.pt'); np.save(out/'validation.npy', pred)
            history.append(row); save(out/f'{phase}_history.json', history)
            print('DEPTH', k, beta, seed, phase, json.dumps(row), flush=True)
        save(out/f'{phase}_history.json', history)
        order_hashes[phase] = order_hash.hexdigest()
        if phase == 'select':
            model.load_state_dict(torch.load(out/'selected.pt', map_location='cuda', weights_only=True))
            np.testing.assert_array_equal(baseval+selection['lambda_value']*corrections(model, vx, seen), np.load(out/'validation.npy'))
            save(out/'selection.json', selection)
        else:
            assert init_hashes[0] == init_hashes[1]
            torch.save(model.state_dict(), out/'final.pt')
            prediction = base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(model, tx, seen)
            np.save(out/'test.npy', prediction)
            model.load_state_dict(torch.load(out/'final.pt', map_location='cuda', weights_only=True))
            np.testing.assert_array_equal(base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(model, tx, seen), prediction)
            save(out/'complete.json', dict(status='PASS', k=k, selection_sha256=digest(out/'selection.json'),
                final_sha256=digest(out/'final.pt'), test_sha256=digest(out/'test.npy'),
                query_sha256=qr['ids_sha256'][str(k)], initial_parameters_sha256=init_hashes[0],
                original_parameters_sha256=init['original_parameters_sha256'],
                trainable_parameters=init['trainable_parameters'], added_parameters=init['added_parameters'],
                full_outer_train_rows=len(data), refit_epochs=epochs, refit_reinitialized=True,
                restored_predictions=True, test_labels_opened=False, own_k_list_only=True,
                shared_old_tensors_preserved=True, batch_order_sha256=order_hashes,
                logical_requests_per_user=1, returned_ids_per_user=k,
                scored_candidates_per_user=m['source_items'],
                select_seconds=sum(h['seconds'] for h in read(out/'select_history.json')),
                refit_seconds=sum(h['seconds'] for h in history)))
        del model, opt, x, y, data, params
    print('DEPTH_FROZEN', k, beta, seed, flush=True)


def evaluate(root):
    check_code(root); frozen = {}; comparisons = []; per_seed = []; budget = []; result = {}
    for k in DEPTHS:
        for beta in BETAS:
            for seed in SEEDS:
                p = root/f'k_{k}/beta_{beta}/seed_{seed}/residual_real'; c = read(p/'complete.json')
                assert c['status'] == 'PASS' and c['test_labels_opened'] is False
                for f, key in [('selection.json','selection_sha256'), ('final.pt','final_sha256'), ('test.npy','test_sha256')]:
                    assert digest(p/f) == c[key]
                assert c['refit_epochs'] == read(p/'selection.json')['epoch']
                assert (p/'selection.json').stat().st_mtime_ns < (p/'final.pt').stat().st_mtime_ns
                peer = read(root/f'k_20/beta_{beta}/seed_{seed}/residual_real/complete.json')
                assert c['initial_parameters_sha256'] == peer['initial_parameters_sha256']
                assert c['batch_order_sha256']['select'] == peer['batch_order_sha256']['select']
                frozen[str(p.relative_to(root))] = digest(p/'complete.json')
    with (root/'evaluation_started.json').open('x') as f: json.dump(frozen, f, indent=2)
    for beta in BETAS:
        cell = PARENT/f'beta_{beta}'; m = read(cell/'manifest.json')
        assert digest(cell/'data/test.csv') == m['hashes']['data/test.csv']
        test = frame(cell/'data/test.csv'); y = test[:,2]; _, users = np.unique(test[:,0], return_inverse=True)
        np.testing.assert_array_equal(test[:,:2], np.load(cell/'data/test_features.npy'))
        mean = predict(dict(np.load(cell/'baseline/statistics.npz')), test[:,1].astype(int), 'item_mean')
        base_metrics = metrics(mean, y, users); result[str(beta)] = {}
        predictions = {'k_100_original': [np.load(cell/f'seed_{s}/residual_real/test.npy') for s in SEEDS]}
        predictions.update({f'k_{k}':[np.load(root/f'k_{k}/beta_{beta}/seed_{s}/residual_real/test.npy') for s in SEEDS] for k in DEPTHS})
        reference = [metrics(p,y,users) for p in predictions['k_100_original']]
        for name, pp in predictions.items():
            rows = []
            for j, seed in enumerate(SEEDS):
                met = metrics(pp[j],y,users); k = int(name.split('_')[1])
                p = (cell/f'seed_{seed}/residual_real' if k==100 else root/f'k_{k}/beta_{beta}/seed_{seed}/residual_real')
                selected = read(p/'selection.json'); c = read(p/'complete.json')
                row = dict(variant=name,beta=beta,seed=seed,k=k,mae=met['mae'],rmse=met['rmse'],
                    delta_mae_vs_original=met['mae']-reference[j]['mae'],delta_rmse_vs_original=met['rmse']-reference[j]['rmse'],
                    epoch=selected['epoch'],lambda_value=selected['lambda_value'],validation_rmse=selected['validation_rmse'])
                per_seed.append(row); rows.append(row)
            a = dict(variant=name,beta=beta,k=k,mae=float(np.mean([r['mae'] for r in rows])),
                mae_std=float(np.std([r['mae'] for r in rows],ddof=1)),rmse=float(np.mean([r['rmse'] for r in rows])),
                rmse_std=float(np.std([r['rmse'] for r in rows],ddof=1)),
                delta_mae_vs_original=float(np.mean([r['delta_mae_vs_original'] for r in rows])),
                delta_rmse_vs_original=float(np.mean([r['delta_rmse_vs_original'] for r in rows])),
                requests_per_user=1,returned_ids_per_user=k,scored_candidates_per_user=m['source_items'])
            comparisons.append(a); result[str(beta)][name]=dict(aggregate=a,per_seed=rows)
        comparisons.append(dict(variant='item_mean',beta=beta,k=0,mae=base_metrics['mae'],mae_std=0.,rmse=base_metrics['rmse'],rmse_std=0.,
            delta_mae_vs_original=base_metrics['mae']-float(np.mean([r['mae'] for r in reference])),
            delta_rmse_vs_original=base_metrics['rmse']-float(np.mean([r['rmse'] for r in reference])),
            requests_per_user=0,returned_ids_per_user=0,scored_candidates_per_user=0))
        for k in [20,50,100,200]:
            budget.append(dict(beta=beta,k=k,overlap_users=m['overlap_users'],test_users=m['users']['test'],
                total_logical_requests=m['overlap_users'],total_returned_ids=m['overlap_users']*k,
                total_scored_candidates=m['overlap_users']*m['source_items'],
                test_logical_requests=m['users']['test'],test_returned_ids=m['users']['test']*k,
                test_scored_candidates=m['users']['test']*m['source_items'],
                convention='hypothetical standalone fixed-K deployment per seed; all target item predictions share one user list'))
    pd.DataFrame(per_seed).to_csv(root/'per_seed.csv', index=False)
    pd.DataFrame(comparisons).to_csv(root/'comparison.csv', index=False)
    pd.DataFrame(budget).to_csv(root/'query_budget.csv', index=False)
    queries = {str(s):read(root/f'queries/seed_{s}/complete.json') for s in SEEDS}
    save(root/'verification.json',dict(status='PASS',final_models=len(frozen),same_parent_splits=True,
        exact_top100_prefixes=all(q['exact_preserved_top100_prefix'] for q in queries.values()),
        all_variants_frozen_before_new_test_labels=True,selection_refit_and_hashes_checked=True,
        shared_initialization_and_minibatches=True,student_teacher_processes_separate=True,
        ordinary_mean_unchanged=True,variants_reported_without_test_selection=True))
    save(root/'summary.json',dict(status='PASS',parent=8650,task=TASK,seeds=SEEDS,betas=BETAS,
        variants=[f'k_{k}' for k in DEPTHS],results=result,queries=queries,
        research_export_cost=dict(logical_requests=sum(q['logical_requests'] for q in queries.values()),
            returned_ids=sum(q['returned_ids'] for q in queries.values()),
            scored_candidates=sum(q['scored_candidates'] for q in queries.values()),
            note='one Top200 export per seed reused across both beta splits and three depths; short lists are offline prefix simulations'),
        caution='exploratory historical test; identical shared old tensors but enlarged common source vocabulary; K200 expands response-depth permission; no active acquisition claim'))
    print('ACQUISITION_EVALUATION_PASS', flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('command', choices=['prepare','export','initialize','train','evaluate'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int);p.add_argument('--beta',type=int);p.add_argument('--k',type=int)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.root)
    elif a.command=='export':export(a.root,a.seed)
    elif a.command=='initialize':initialize(a.root,a.seed)
    elif a.command=='train':train(a.root,a.seed,a.beta,a.k)
    else:evaluate(a.root)


if __name__=='__main__':main()
