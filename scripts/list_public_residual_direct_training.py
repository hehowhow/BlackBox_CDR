"""Direct-execution copy of the frozen grouped-user training loop.

Only the scheduler guard and execution receipt change. Guard/hash delegates
remain attached to the study's process-local model and access audits.
verify_training_equivalence() checks the complete training-function source
against the archived function after precisely these substitutions.
"""
import hashlib
import inspect
import json
import os
import time
import numpy as np
import torch

import list_coarse_sid_study as archived
from list_coarse_sid_study import (
    check_code, read, output, save, frame, release_dir, digest, predict,
    seed_all, ResidualList, GroupedRatings, correction,
)


def require_direct_authorization(root):
    record = read(root / 'direct_authorization.json')
    assert record['mode'] == 'direct' and record['authorized_gpus'] == 2
    assert os.environ.get('LIST_EXECUTION_MODE') == 'direct'
    assert not os.environ.get('SLURM_JOB_ID'), 'Direct runs must not claim a Slurm job'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')


def direct_receipt(root):
    return dict(execution_backend='direct', job=None, pid=os.getpid(),
                gpu=os.environ['CUDA_VISIBLE_DEVICES'], time=time.time(),
                authorization_sha256=digest(root / 'direct_authorization.json'))


def train(root, task, variant, beta, seed):
    from list_coarse_sid_model import make_model
    check_code(root)
    require_direct_authorization(root)
    cfg = read(root / 'protocol.json')
    out = output(root, task, variant, beta, seed)
    opened = archived.target_guard(root, variant, out)
    save(out / 'started.json', direct_receipt(root))
    cell = root / 'data' / task / f'beta_{beta}'
    manifest = read(cell / 'manifest.json')
    validation = frame(cell / 'data/validation.csv')
    vx = torch.as_tensor(validation[:, :2], device='cuda', dtype=torch.long)
    tx = torch.as_tensor(np.load(cell / 'data/test_features.npy'), device='cuda', dtype=torch.long)
    kind, family = variant.split('_')
    selection, initials, phases = None, [], {}
    for phase in ['select', 'refit']:
        release = release_dir(root, task, beta, seed, phase)
        released = read(release / 'release.json')
        path = release / f'{kind}.npz'
        assert digest(path) == released[f'{kind}_sha256']
        tokens = np.load(path)['item_ids']
        vocab_size = released[f'{kind}_vocab_size']
        lists = torch.zeros((manifest['target_users'], cfg['k']), device='cuda', dtype=torch.long)
        lists[:len(tokens)] = torch.as_tensor(tokens, device='cuda')
        data_path = cell / ('data/inner_train.csv' if phase == 'select' else 'data/train.csv')
        assert digest(data_path) == manifest['hashes'][str(data_path.relative_to(cell))]
        data = frame(data_path)
        assert not np.intersect1d(data[:, 0], np.load(cell / 'data/test_users.npy')).size
        if phase == 'select':
            assert not np.intersect1d(data[:, 0], validation[:, 0]).size
        bpath = cell / ('inner_baseline' if phase == 'select' else 'baseline')
        base64 = predict(dict(np.load(bpath / 'statistics.npz')), np.arange(manifest['target_items']), 'item_mean')
        base = torch.as_tensor(base64.astype(np.float32), device='cuda')
        seen = torch.as_tensor(np.load(bpath / 'seen.npy'), device='cuda')
        seed_all(seed)
        reference = ResidualList(lists, released['raw_vocab_size'], manifest['target_items'], base, cfg).cuda()
        common = {n: p.detach().clone() for n, p in reference.named_parameters() if not n.startswith('source_items.')}
        del reference
        seed_all(seed)
        model = make_model(lists, vocab_size, manifest['target_items'], base, cfg, family).cuda()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in common:
                    p.copy_(common[name])
        common_hash = hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in common.values())).hexdigest()
        del common
        initials.append(archived.parameter_hash(model))
        assert len(initials) == 1 or initials[0] == initials[1], 'Refit must use same parameter initialization'
        grouped = GroupedRatings(data, manifest['overlap_users'], 'cuda')
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(seed + 200)
        seed_all(seed + 100)
        if phase == 'select':
            baseval = base64[validation[:, 1].astype(int)]
            selection = dict(epoch=0, lambda_value=0., validation_rmse=float(np.sqrt(np.square(baseval-validation[:, 2]).mean())))
            torch.save(model.state_dict(), out / 'selected.pt')
            np.save(out / 'validation.npy', baseval)
        history = []
        epochs = cfg['epochs'] if phase == 'select' else selection['epoch']
        for epoch in range(1, epochs + 1):
            tick = time.monotonic()
            model.train()
            total, updates = 0., 0
            for batch_id, index in enumerate(torch.randperm(len(grouped.users), device='cuda', generator=generator).split(cfg['user_batch_size'])):
                batch = grouped.batch(index, base)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = model.objective(*batch, seed + epoch * 10000 + batch_id)
                assert torch.isfinite(loss), (task, variant, phase, epoch)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * int(batch[-1].sum())
                updates += 1
            row = dict(epoch=epoch, objective=total/grouped.rows, updates=updates, seconds=time.monotonic()-tick)
            if phase == 'select':
                delta = correction(model, vx, seen, seed)
                row['validation'] = {}
                for lam in cfg['lambdas']:
                    pred = baseval + lam * delta
                    score = float(np.sqrt(np.square(pred-validation[:, 2]).mean()))
                    row['validation'][str(lam)] = score
                    if score < selection['validation_rmse']:
                        selection = dict(epoch=epoch, lambda_value=lam, validation_rmse=score)
                        torch.save(model.state_dict(), out / 'selected.pt')
                        np.save(out / 'validation.npy', pred)
            history.append(row)
            save(out / f'{phase}_history.json', history)
            print('COARSE_EPOCH', task, beta, seed, variant, phase, json.dumps(row), flush=True)
        save(out / f'{phase}_history.json', history)
        phases[phase] = dict(common_initial_sha256=common_hash, initial_sha256=initials[-1],
            release_sha256=released[f'{kind}_sha256'], vocab_size=vocab_size,
            overlap_train_rows=grouped.rows, grouped_users=len(grouped.users),
            full_target_train_rows=len(data), epochs=epochs,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            fit_users_sha256=released['fit_users_sha256'])
        if phase == 'select':
            model.load_state_dict(torch.load(out / 'selected.pt', map_location='cuda', weights_only=True))
            np.testing.assert_allclose(baseval+selection['lambda_value']*correction(model,vx,seen,seed), np.load(out/'validation.npy'), atol=1e-7, rtol=0)
            save(out / 'selection.json', selection)
        else:
            torch.save(model.state_dict(), out / 'final.pt')
            pred = base64[tx[:, 1].cpu().numpy()] + selection['lambda_value']*correction(model,tx,seen,seed)
            np.save(out / 'test.npy', pred)
            model.load_state_dict(torch.load(out / 'final.pt', map_location='cuda', weights_only=True))
            np.testing.assert_array_equal(pred, base64[tx[:, 1].cpu().numpy()]+selection['lambda_value']*correction(model,tx,seen,seed))
        del model, optimizer, grouped, lists, data
    save(out / 'access_audit.json', dict(status='PASS', opened=sorted(opened), enforced_interface=kind,
        trusted_mapping_blocked=True, raw_queries_blocked=True, test_labels_blocked=True))
    save(out / 'complete.json', dict(status='PASS', task=task, variant=variant, beta=beta, seed=seed,
        phases=phases, test_labels_opened=False, restored_predictions=True, refit_reinitialized=True,
        final_sha256=digest(out/'final.pt'), test_sha256=digest(out/'test.npy'),
        selection_sha256=digest(out/'selection.json'), finished=time.time()))
    print('COARSE_FROZEN', task, beta, seed, variant, flush=True)


def verify_training_equivalence():
    original = inspect.getsource(archived.train).strip()
    expected = original.replace(
        "assert os.environ.get('SLURM_JOB_ID'), 'Training must run under Slurm'",
        "require_direct_authorization(root)").replace(
        "save(out / 'started.json', dict(job=os.environ['SLURM_JOB_ID'], gpu=os.environ['CUDA_VISIBLE_DEVICES'], time=time.time()))",
        "save(out / 'started.json', direct_receipt(root))").replace(
        "opened = target_guard(root, variant, out)",
        "opened = archived.target_guard(root, variant, out)").replace(
        "initials.append(parameter_hash(model))",
        "initials.append(archived.parameter_hash(model))")
    actual = inspect.getsource(train).strip()
    assert actual == expected, 'Direct runner changed archived learning logic'
    return dict(status='PASS', archived_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
                direct_function_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                differences=['scheduler guard', 'execution receipt', 'audit delegate references'])
