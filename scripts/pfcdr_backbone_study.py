"""PFCDR MF/GMF/DNN comparison with frozen cold-user validation protocol."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import softmax
from book_movie_protocol import BASE, digest, save, predict
from residual_experiment import guard_test_labels, seed_all
from residual_summary import metrics, paired_interval

UP = BASE/'external/PFCDR'
PRIOR = BASE/'runs/residual_study_20260922/book_movie'
BACKBONES = ['MF', 'GMF', 'DNN']
cwd = Path.cwd()
os.chdir(UP)
sys.path.insert(0, str(UP))
from models import MFBasedModel, GMFBasedModel, DNNBasedModel
os.chdir(cwd)


def read(path):
    return json.loads(path.read_text())


def make_model(name, manifest, dim=10):
    src = dict(uid_src=manifest['source_users'], iid_src=manifest['source_items'])
    tgt = dict(uid_tgt=manifest['target_users'], iid_tgt=manifest['target_items'])
    if name == 'MF': return MFBasedModel(src, tgt, 2, dim, 70000).cuda()
    if name == 'GMF': return GMFBasedModel(src, tgt, 2, dim).cuda()
    return DNNBasedModel(src, tgt, 2, dim, 70000).cuda()


def table(model, name, domain):
    module = getattr(model, domain+'_model')
    return module if name == 'MF' else module.embedding


def effective_users(model, name, domain, ids=None):
    t = table(model, name, domain)
    u = t.uid_embedding.weight if ids is None else t.uid_embedding(ids)
    module = getattr(model, domain+'_model')
    if name == 'GMF': return u*module.linear.weight[0]
    if name == 'DNN': return module.linear(u)
    return u


@torch.no_grad()
def voted_vectors(model, name, prototypes, ks, batch=32):
    raw = table(model, name, 'tgt').uid_embedding.weight
    eff = effective_users(model, name, 'tgt')
    # Preserve the published GMF matcher: it uses raw U, not U * linear.weight.
    match = raw if name == 'GMF' else eff
    result = {k: [] for k in ks}
    for p in model.rp_mapping(prototypes).split(batch):
        order = torch.sort((match@p.T-5).abs(), dim=0).indices
        for k in ks:
            result[k].append(eff[order[:k].T].mean(1))
    return {k: torch.cat(v) for k, v in result.items()}


@torch.no_grad()
def rating_predictions(model, name, vectors, ids, x, fallback=None, seen=None):
    lookup = torch.full((table(model, name, 'tgt').uid_embedding.num_embeddings,), -1, device='cuda', dtype=torch.long)
    lookup[ids] = torch.arange(len(ids), device='cuda')
    idx = lookup[x[:, 0]]
    assert (idx >= 0).all()
    p = (vectors[idx]*table(model, name, 'tgt').iid_embedding(x[:, 1])).sum(1)
    if fallback is not None:
        p = torch.where(seen[x[:, 1]], p, fallback[x[:, 1]])
    return p.cpu().numpy()


def prepare(root):
    root.mkdir(exist_ok=False, parents=True)
    (root/'data').symlink_to(PRIOR/'data', target_is_directory=True)
    (root/'baseline').symlink_to(PRIOR/'baseline', target_is_directory=True)
    manifest = read(PRIOR/'manifest.json')
    save(root/'manifest.json', manifest)
    files = ['pfcdr_backbone_study.py', 'pfcdr_backbone_verify.py', 'book_movie_protocol.py',
             'residual_experiment.py', 'residual_summary.py']
    cfg = dict(task='Book->Movie', backbones=BACKBONES, seed=2020, dim=10,
               pretrain_epochs=10, pretrain_lr=.01, pretrain_weight_decay=0., pretrain_batch=512,
               inversion_iterations=300, inversion_batch=200, inversion_lr=1e-4,
               inversion_weight_decay=1e-4, inversion_target=5, inversion_warmup=10,
               bridge_epochs=10, bridge_lr=.005, bridge_batch=512, bridge_weight_decay=0.,
               primary_k=70000, reported_ks=[50000, 70000], selection='validation_rmse',
               primary_selection='Minimum validation RMSE among bridge epochs 1..10 independently for each predeclared K',
               source_target_pretraining='Fixed final epoch; no target validation cold-user vectors used for pretrain selection',
               voting_candidates='All target model rows, including untrained validation/test user rows, matching public code',
               gmf_matching='Public raw target user vector matching; output uses learned GMF linear weights',
               unknown_target_items='Only items absent from target training fall back to frozen smoothed mean; also save raw predictions',
               initialization='Matched source/target embedding initialization across backbones; separate deterministic head and mapping RNG',
               inversion_implementation='Cache per-user item scores; optimize full catalog weights with Adam, same loss/schedule and post-step best-input convention as public inversion',
               split='Existing 60/20/20 overlap-user split; only explicit train_users enter target inversion/bridge',
               scope='One seed, no learning-rate search, exploratory backbone screening; not ten-run paper reproduction',
               test_policy='Test IDs may receive source prototypes; test target labels opened only after all three runs freeze selections',
               bootstrap_repeats=2000, bootstrap_seed=991,
               code_sha256={str(BASE/'scripts'/p): digest(BASE/'scripts'/p) for p in files},
               upstream_sha256={str(UP/p): digest(UP/p) for p in ['models.py', 'inversion.py', 'float_embedding.py', 'inversion_config.json']})
    save(root/'protocol.json', cfg)
    for name in BACKBONES: (root/name).mkdir()
    print('PREPARED', root, flush=True)


def selftest(root):
    checks = []
    for name in BACKBONES:
        seed_all(731)
        m = make_model(name, dict(source_users=8, source_items=31, target_users=70003, target_items=19))
        for p in m.parameters(): p.requires_grad_(False)
        for domain, ni in [('src', 31), ('tgt', 19)]:
            u = torch.arange(4, device='cuda')
            x = torch.stack([u, torch.arange(4, device='cuda')], dim=1)
            native = m(x, 'train_'+domain)
            optimized = (effective_users(m, name, domain, u)*table(m, name, domain).iid_embedding(x[:, 1])).sum(1)
            torch.testing.assert_close(native, optimized, rtol=2e-5, atol=2e-5)
            w = torch.softmax(torch.randn(4, ni, device='cuda'), 1).requires_grad_()
            inputs = torch.cat([u[:, None].float(), w], 1).unsqueeze(-1)
            native = m(inputs, domain+'_inversion')
            gn = torch.autograd.grad((native-5).square().mean(), w)[0]
            coefficients = effective_users(m, name, domain, u)@table(m, name, domain).iid_embedding.weight.T
            fast = (w*coefficients).sum(1)
            gf = torch.autograd.grad((fast-5).square().mean(), w)[0]
            torch.testing.assert_close(native, fast, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(gn, gf, rtol=3e-5, atol=3e-5)
            checks.append(dict(backbone=name, domain=domain, integer_forward=True, inversion_forward=True, inversion_gradient=True))
        prototypes = torch.randn(3, 10, device='cuda')
        k = 50000 if name == 'GMF' else 70000
        vectors = voted_vectors(m, name, prototypes, [k])
        items = torch.arange(3, device='cuda')
        native = m(torch.cat([items[:, None].float(), prototypes], 1), 'test_source_free')
        cached = (vectors[k]*table(m, name, 'tgt').iid_embedding(items)).sum(1)
        torch.testing.assert_close(native, cached, rtol=3e-5, atol=3e-5)
        checks.append(dict(backbone=name, native_voting=True, k=k))
        del m
        torch.cuda.empty_cache()
    save(root/'selftest.json', dict(status='PASS', checks=checks))
    print('BACKBONE_SELFTEST_PASS', flush=True)


def initialize(model, name, seed):
    hashes = {}
    for offset, domain in enumerate(['src', 'tgt']):
        seed_all(seed+1000+offset)
        t = table(model, name, domain)
        nn.init.normal_(t.uid_embedding.weight)
        nn.init.normal_(t.iid_embedding.weight)
        import hashlib
        hashes[domain] = hashlib.sha256(t.uid_embedding.weight.detach().cpu().numpy().tobytes()+
                                        t.iid_embedding.weight.detach().cpu().numpy().tobytes()).hexdigest()
        if name != 'MF':
            seed_all(seed+2000+offset)
            getattr(model, domain+'_model').linear.reset_parameters()
    seed_all(seed+3000)
    model.rp_mapping.reset_parameters()
    return hashes


def pretrain(root, out, model, name, cfg):
    history = []
    for offset, (domain, filename) in enumerate([('src', 'source'), ('tgt', 'train')]):
        data = pd.read_csv(root/f'data/{filename}.csv', header=None).to_numpy()
        x = torch.tensor(data[:, :2], dtype=torch.long, device='cuda')
        y = torch.tensor(data[:, 2], dtype=torch.float32, device='cuda')
        module = getattr(model, domain+'_model')
        optimizer = torch.optim.Adam(module.parameters(), lr=cfg['pretrain_lr'], weight_decay=cfg['pretrain_weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(cfg['seed']+4000+offset)
        for epoch in range(1, cfg['pretrain_epochs']+1):
            tick = time.monotonic()
            total = torch.zeros((), device='cuda')
            model.train()
            for idx in torch.randperm(len(x), generator=generator, device='cuda').split(cfg['pretrain_batch']):
                optimizer.zero_grad(set_to_none=True)
                loss = (model(x[idx], 'train_'+domain)-y[idx]).square().mean()
                loss.backward()
                optimizer.step()
                total += loss.detach()*len(idx)
            row = dict(domain=domain, epoch=epoch, mse=total.item()/len(x), seconds=time.monotonic()-tick)
            assert np.isfinite(row['mse'])
            history.append(row)
            save(out/'pretrain_history.json', history)
            print('PRETRAIN', name, json.dumps(row), flush=True)
        del data, x, y, optimizer
    torch.save(model.state_dict(), out/'pretrained.pt')


def invert(root, out, model, name, cfg):
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    train_ids = np.load(root/'data/train_users.npy')
    n = read(root/'manifest.json')['overlap_users']
    for offset, (domain, ids) in enumerate([('src', np.arange(n)), ('tgt', train_ids)]):
        rng = np.random.RandomState(cfg['seed']+5000+offset)
        item_vectors = table(model, name, domain).iid_embedding.weight.detach()
        all_proto, records = [], []
        for start in range(0, len(ids), cfg['inversion_batch']):
            tick = time.monotonic()
            uid = torch.tensor(ids[start:start+cfg['inversion_batch']], dtype=torch.long, device='cuda')
            with torch.no_grad():
                coeff = effective_users(model, name, domain, uid)@item_vectors.T
            initial = softmax(rng.randn(len(uid), len(item_vectors)), axis=1).astype(np.float32)
            weights = torch.tensor(initial, device='cuda', requires_grad=True)
            optimizer = torch.optim.Adam([weights], lr=cfg['inversion_lr'], weight_decay=cfg['inversion_weight_decay'])
            best_loss, best = float('inf'), None
            initial_loss = None
            for iteration in range(cfg['inversion_iterations']):
                if iteration < cfg['inversion_warmup']:
                    lr = cfg['inversion_lr']*(iteration+1)/cfg['inversion_warmup']
                else:
                    lr = .5*(1+np.cos(np.pi*(iteration-cfg['inversion_warmup'])/
                                     (cfg['inversion_iterations']-cfg['inversion_warmup'])))*cfg['inversion_lr']
                optimizer.param_groups[0]['lr'] = lr
                p = (weights*coeff).sum(1)
                loss = (p-cfg['inversion_target']).square().mean()
                value = loss.item()
                if initial_loss is None: initial_loss = value
                assert np.isfinite(value)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # The public module associates pre-step loss with post-step inputs.
                if value < best_loss or iteration == 1:
                    best = weights.detach().clone()
                    best_loss = value
            with torch.no_grad():
                proto = best@item_vectors
                score = (effective_users(model, name, domain, uid)*proto).sum(1)
                actual = (score-cfg['inversion_target']).square().mean().item()
            all_proto.append(proto.cpu().numpy())
            records.append(dict(start=start, users=len(uid), iterations=cfg['inversion_iterations'],
                                initial_mse=initial_loss, selected_pre_step_mse=best_loss,
                                saved_prototype_mse=actual, seconds=time.monotonic()-tick))
            save(out/f'inversion_{domain}_history.json', records)
            print('INVERSION', name, domain, start+len(uid), '/', len(ids), 'mse', actual,
                  'seconds', records[-1]['seconds'], flush=True)
            del weights, optimizer, best, coeff
        np.savez(out/f'prototypes_{domain}.npz', user_ids=ids, prototypes=np.concatenate(all_proto))
        torch.cuda.empty_cache()


def bridge(root, out, model, name, cfg):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.rp_mapping.parameters(): p.requires_grad_(True)
    src, tgt = np.load(out/'prototypes_src.npz'), np.load(out/'prototypes_tgt.npz')
    train_ids = np.load(root/'data/train_users.npy')
    val_ids = np.load(root/'data/validation_users.npy')
    test_ids = np.load(root/'data/test_users.npy')
    np.testing.assert_array_equal(src['user_ids'], np.arange(len(src['user_ids'])))
    np.testing.assert_array_equal(tgt['user_ids'], train_ids)
    assert not set(train_ids)&set(val_ids) and not set(train_ids)&set(test_ids)
    sp = torch.tensor(src['prototypes'], device='cuda')
    tp = torch.tensor(tgt['prototypes'], device='cuda')
    tids = torch.tensor(train_ids, device='cuda')
    vids = torch.tensor(val_ids, device='cuda')
    val = pd.read_csv(root/'data/validation.csv', header=None).to_numpy()
    vx = torch.tensor(val[:, :2], dtype=torch.long, device='cuda')
    base = torch.tensor(np.load(root/'baseline/values.npy'), device='cuda')
    seen = torch.tensor(np.load(root/'baseline/seen.npy'), device='cuda')
    optimizer = torch.optim.Adam(model.rp_mapping.parameters(), lr=cfg['bridge_lr'], weight_decay=cfg['bridge_weight_decay'])
    generator = torch.Generator(device='cuda').manual_seed(cfg['seed']+6000)
    selected = {k: dict(validation_rmse=float('inf')) for k in cfg['reported_ks']}
    history = []
    for epoch in range(1, cfg['bridge_epochs']+1):
        tick = time.monotonic()
        model.train()
        total = torch.zeros((), device='cuda')
        for idx in torch.randperm(len(train_ids), generator=generator, device='cuda').split(cfg['bridge_batch']):
            optimizer.zero_grad(set_to_none=True)
            loss = (model.rp_mapping(sp[tids[idx]])-tp[idx]).square().mean()
            loss.backward(); optimizer.step()
            total += loss.detach()*len(idx)
        model.eval()
        vectors = voted_vectors(model, name, sp[vids], cfg['reported_ks'])
        row = dict(epoch=epoch, mapping_mse=total.item()/len(train_ids), validation={})
        for k in cfg['reported_ks']:
            pred = rating_predictions(model, name, vectors[k], vids, vx, base, seen)
            rmse = float(np.sqrt(np.square(pred-val[:, 2]).mean()))
            mae = float(np.abs(pred-val[:, 2]).mean())
            assert np.isfinite(rmse)
            row['validation'][str(k)] = dict(rmse=rmse, mae=mae)
            if rmse < selected[k]['validation_rmse']:
                torch.save(model.rp_mapping.state_dict(), out/f'bridge_k{k}.pt')
                np.save(out/f'validation_k{k}.npy', pred)
                selected[k] = dict(epoch=epoch, validation_rmse=rmse, validation_mae=mae)
        row['seconds'] = time.monotonic()-tick
        history.append(row)
        save(out/'bridge_history.json', history)
        print('BRIDGE', name, json.dumps(row), flush=True)
    # Restore every validation-selected model and verify its predictions before testing.
    tx = torch.tensor(np.load(root/'data/test_features.npy'), dtype=torch.long, device='cuda')
    xids = torch.tensor(test_ids, device='cuda')
    for k in cfg['reported_ks']:
        model.rp_mapping.load_state_dict(torch.load(out/f'bridge_k{k}.pt', weights_only=True, map_location='cuda'))
        vv = voted_vectors(model, name, sp[vids], [k])[k]
        pred = rating_predictions(model, name, vv, vids, vx, base, seen)
        np.testing.assert_array_equal(pred, np.load(out/f'validation_k{k}.npy'))
        tv = voted_vectors(model, name, sp[xids], [k])[k]
        raw = rating_predictions(model, name, tv, xids, tx)
        final = np.where(seen[tx[:, 1]].cpu().numpy(), raw, base[tx[:, 1]].cpu().numpy())
        np.save(out/f'test_raw_k{k}.npy', raw)
        np.save(out/f'test_k{k}.npy', final)
        np.save(out/f'test_vectors_k{k}.npy', tv.cpu().numpy())
        selected[k].update(checkpoint_sha256=digest(out/f'bridge_k{k}.pt'),
                           validation_sha256=digest(out/f'validation_k{k}.npy'),
                           test_sha256=digest(out/f'test_k{k}.npy'), raw_test_sha256=digest(out/f'test_raw_k{k}.npy'))
    save(out/'selection.json', selected)
    print('BACKBONE_SELECTION_FROZEN', name, flush=True)


def run(root, name):
    cfg, manifest = read(root/'protocol.json'), read(root/'manifest.json')
    for p, value in {**cfg['code_sha256'], **cfg['upstream_sha256']}.items(): assert digest(Path(p)) == value
    guard_test_labels(root)
    for p in ['data/source.csv', 'data/train.csv', 'data/validation.csv', 'data/train_users.npy',
              'data/validation_users.npy', 'data/test_users.npy', 'data/test_features.npy']:
        assert digest(root/p) == manifest['hashes'][p]
    out = root/name
    assert not (out/'started.json').exists()
    save(out/'started.json', dict(job=os.environ.get('SLURM_JOB_ID'), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 gpu_name=torch.cuda.get_device_name(), torch=torch.__version__, seed=cfg['seed']))
    seed_all(cfg['seed'])
    model = make_model(name, manifest, cfg['dim'])
    save(out/'initialization.json', initialize(model, name, cfg['seed']))
    pretrain(root, out, model, name, cfg)
    invert(root, out, model, name, cfg)
    bridge(root, out, model, name, cfg)
    save(out/'complete.json', dict(status='PASS', pretrained_sha256=digest(out/'pretrained.pt'),
                                   prototypes={d:digest(out/f'prototypes_{d}.npz') for d in ['src', 'tgt']},
                                   selection_sha256=digest(out/'selection.json')))


def evaluate(root):
    cfg, manifest = read(root/'protocol.json'), read(root/'manifest.json')
    assert read(root/'verification.json')['status'] == 'PASS'
    complete = {n: read(root/n/'complete.json') for n in BACKBONES}
    with (root/'evaluation_started.json').open('x') as f: json.dump(complete, f, indent=2)
    for p, h in {**manifest['hashes'], **manifest['baseline_hashes']}.items(): assert digest(root/p) == h
    test = pd.read_csv(root/'data/test.csv', header=None).to_numpy()
    np.testing.assert_array_equal(test[:, :2], np.load(root/'data/test_features.npy'))
    y = test[:, 2]
    _, users = np.unique(test[:, 0], return_inverse=True)
    stats = dict(np.load(root/'baseline/statistics.npz'))
    baselines = {n: predict(stats, test[:, 1].astype(int), n) for n in ['item_mean', 'item_median', 'smoothed_mean_5']}
    results = dict(baselines={k:metrics(p,y,users) for k,p in baselines.items()}, models={})
    rows=[]
    for n in BACKBONES:
        out = root/n
        assert digest(out/'selection.json') == complete[n]['selection_sha256']
        selected = read(out/'selection.json')
        results['models'][n]={}
        for k in cfg['reported_ks']:
            assert digest(out/f'test_k{k}.npy') == selected[str(k)]['test_sha256']
            p, raw = np.load(out/f'test_k{k}.npy'), np.load(out/f'test_raw_k{k}.npy')
            m = metrics(p,y,users)
            refs = {**baselines, 'MF_same_k':np.load(root/'MF'/f'test_k{k}.npy')}
            comparisons = {key:paired_interval(p[None,:],other[None,:],y,users,cfg) for key,other in refs.items()}
            for c in comparisons.values(): c['interpretation']='Negative favors model; user-cluster CI conditional on one fitted seed, not training-seed uncertainty'
            results['models'][n][str(k)] = dict(selection=selected[str(k)], metrics=m, raw_metrics=metrics(raw,y,users), comparisons=comparisons)
            rows.append(dict(backbone=n,k=k,epoch=selected[str(k)]['epoch'],validation_rmse=selected[str(k)]['validation_rmse'],mae=m['mae'],rmse=m['rmse']))
    save(root/'summary.json', dict(status='PASS', protocol=cfg, **results))
    with (root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Book→Movie PFCDR 基础模型比较','','单种子 2020；验证 RMSE 选映射 epoch；主比较固定 K=70000。',
           '相同 60/20/20 划分、相同均值基线、预训练 10 轮、反演 300 次、映射 10 轮。',
           '验证/测试未见目标物品统一回退平滑均值，原始无回退指标也保存于 summary.json。','',
           '| 方法 | K | 选定 epoch | MAE | RMSE |','| --- | ---: | ---: | ---: | ---: |']
    for n,m in results['baselines'].items(): lines.append(f"| {n} | — | — | {m['mae']:.6f} | {m['rmse']:.6f} |")
    for r in rows: lines.append(f"| PFCDR-{r['backbone']} | {r['k']} | {r['epoch']} | {r['mae']:.6f} | {r['rmse']:.6f} |")
    lines+=['','完整配对区间、原始预测指标、配置和校验见 summary.json、verification.json。',
            '本轮仅筛查基础模型，不搜索学习率，也不构成十次重复的论文复现。',
            '保留官方候选池（包括未训练冷用户行）及 GMF 原始用户向量匹配方式；不应将其理解成唯一合理的 GMF 匹配规则。',
            '历史测试集此前已使用，当前结果是探索性结果。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('BACKBONE_STUDY_PASS', json.dumps(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','selftest','run','evaluate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--backbone', choices=BACKBONES)
    a=parser.parse_args()
    if a.stage=='run': run(a.root.resolve(),a.backbone)
    else: globals()[a.stage](a.root.resolve())
