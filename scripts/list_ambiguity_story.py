"""Music->Movie LIST conditional-transfer ambiguity study, two Slurm GPUs."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from book_movie_protocol import BASE, digest, save, predict
from list_source_backbones import check_code, read, student_guard
from residual_experiment import seed_all
from small_task_comparison import PRIOR, TASK, SEEDS, BETAS, ResidualList, frame
from list_ambiguity_model import ConditionalList, VARIANTS, langevin

PARENT = BASE / 'runs/small_task_comparison_8650'


def parameter_hash(model, common=False):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if not common or not name.startswith(('log_variance.', 'energy_net.')):
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    previous = read(PARENT / 'protocol.json')
    assert read(PARENT / 'verification.json')['status'] == 'PASS'
    for path, expected in previous['code_hashes'].items():
        assert digest(Path(path)) == expected, path
    files = [Path(__file__), BASE / 'scripts/list_ambiguity_model.py',
             BASE / 'scripts/verify_list_ambiguity.py', BASE / 'scripts/slurm_list_ambiguity.sh']
    cfg = dict(previous, parent=str(PARENT), parent_protocol_sha256=digest(PARENT / 'protocol.json'),
        variants=VARIANTS, arms=['real'], user_batch_size=64, latent_std=.3,
        logvar_bounds=[math.log(.05 ** 2), math.log(2. ** 2)], energy_bound=5.,
        train_chains=32, eval_chains=128, mala_step=.25, eval_user_batch=128,
        observation_variance=1.,
        unchanged='DNN Top100 IDs/ranks; rank pooling; original shared parameter initialization; 32-D latent and target item dot; training-only ordinary item means; 10 epochs; Adam .001/wd1e-5; cold validation and fresh refit',
        batching='64 complete overlap training users per update, every observed rating retained; sum user joint losses / number of ratings in batch; dropout once per user; identical point_matched control',
        target_only='target-only rows contribute to fixed ordinary means; zero residual gives no learned likelihood gradient; excluded from grouped updates in ALL new variants',
        conditional_gaussian='z|h ~ N(original_MLP(h),diag(var(h))); fixed .3 std or learned variance; exact joint observed-rating marginal likelihood; no KL penalty',
        conditional_energy='Gaussian conditional reference times exp(-bounded_MLP(z,h)); Gaussian confinement; joint ML gradient estimated by prior and training-posterior MALA; only observed training ratings enter posterior',
        sampling='whitened preconditioned MALA .25; 32 paired chains for train prior/posterior with common random draws; 128 paired chains for cold inference; 20/60 steps train and inference; no differentiation through chain; no claim of complete mixing',
        prediction='mean latent (analytic Gaussian / sampled EBM) dot target vector + ordinary mean; validation-selected global lambda; no rating clipping or per-user gate',
        selection='epoch0..10 and lambda0/.25/.5/.75/1 by cold validation RMSE, then fresh outer refit for selected epochs',
        evaluation='all30 new models/predictions frozen before opening test labels; historical split reused, exploratory; average individual seed metrics, not seed ensemble',
        literature={'latent_EBM_NeurIPS2020': 'https://papers.nips.cc/paper_files/paper/2020/hash/fa3060edb66e6ff4507886f9912e1ab9-Abstract.html'},
        code_hashes={**previous['code_hashes'], **{str(p.resolve()): digest(p) for p in files}})
    save(root / 'protocol.json', cfg)
    for beta in BETAS:
        cell = root / f'beta_{beta}'
        cell.mkdir()
        old = PARENT / f'beta_{beta}'
        manifest = read(old / 'manifest.json')
        assert manifest['cross_domain_raw_item_overlap'] == 0 and manifest['overlap_users'] == 17904
        for path, expected in manifest['hashes'].items():
            if path != 'data/test.csv':
                assert digest(old / path) == expected
        for name in ['data', 'baseline', 'inner_baseline']:
            (cell / name).symlink_to((old / name).resolve(), target_is_directory=True)
        save(cell / 'manifest.json', manifest)
    tasks = []
    for variant in VARIANTS:
        for beta in BETAS:
            for seed in SEEDS:
                out = root / variant / f'beta_{beta}/seed_{seed}'
                out.mkdir(parents=True)
                tasks.append(dict(variant=variant, beta=beta, seed=seed))
    save(root / 'tasks.json', tasks)
    save(root / 'data_verification.json', dict(status='PASS', same_parent_splits=True,
        deduplicated=True, no_test_labels_opened=True, all_variants_fixed=True))
    print('AMBIGUITY_PREPARED', root, flush=True)


def selftest(root, device):
    check_code(root)
    student_guard(root)
    cfg = read(root / 'protocol.json')
    seed_all(123)
    lists = torch.tensor([[1, 2, 3], [3, 2, 1], [0, 0, 0]], device=device)
    base = torch.tensor([3., 4., 2.], device=device)
    users = torch.arange(3, device=device)
    seed_all(123)
    original = ResidualList(lists, 3, 3, base, cfg).to(device)
    hashes = []
    for variant in VARIANTS:
        seed_all(123)
        model = ConditionalList(lists, 3, 3, base, cfg, variant).to(device)
        hashes.append(parameter_hash(model))
        assert parameter_hash(model, True) == parameter_hash(original)
        model.eval()
        torch.testing.assert_close(model.user_mean(users, 222), torch.zeros(3, 32, device=device), atol=1e-7, rtol=0)
        with torch.no_grad():
            model.encoder[-1].weight.normal_(std=.1)
        items = torch.tensor([[0, 1, 2], [1, 2, 0]], device=device)
        mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]], device=device)
        residual = torch.tensor([[1., -.3, .5], [-.2, 1., 0.]], device=device)
        objective, _ = model.objective(users[:2], items, residual, mask, 456)
        objective.backward()
        assert torch.isfinite(objective)
        assert model.encoder[-1].weight.grad.abs().sum() > 0
        assert model.target_items.weight.grad.abs().sum() > 0
        assert model.source_items.weight.grad.abs().sum() > 0
        if variant in {'gaussian_learned', 'energy_20', 'energy_60'}:
            assert model.log_variance.bias.grad.abs().sum() > 0
        if variant.startswith('energy'):
            assert model.energy_net[-1].weight.grad.abs().sum() > 0
        assert model.user_mean(users, 789)[-1].abs().max() == 0
    assert len(set(hashes)) == 1

    # Exact Gaussian integral against an independently formed rating covariance.
    generator = torch.Generator(device=device).manual_seed(987)
    v = torch.randn((2, 5, 3), device=device, dtype=torch.float64, generator=generator)
    r = torch.randn((2, 5), device=device, dtype=torch.float64, generator=generator)
    mean = torch.randn((2, 3), device=device, dtype=torch.float64, generator=generator).requires_grad_()
    variance = torch.tensor([[.1, .4, .7], [.6, .2, .3]], device=device, dtype=torch.float64)
    a, b, c = v.transpose(1, 2) @ v, (v * r[..., None]).sum(1), r.square().sum(1)
    actual = ConditionalList.gaussian_nll_twice(mean, variance, a, b, c)
    covariance = torch.eye(5, device=device, dtype=torch.float64) + (v * variance[:, None]) @ v.transpose(1, 2)
    reference = -2 * torch.distributions.MultivariateNormal((v @ mean[..., None]).squeeze(-1), covariance_matrix=covariance).log_prob(r) - 5 * math.log(2 * math.pi)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
    ga, = torch.autograd.grad(actual.sum(), mean, retain_graph=True)
    gr, = torch.autograd.grad(reference.sum(), mean)
    torch.testing.assert_close(ga, gr, atol=1e-10, rtol=1e-10)

    # Linear tilt of a Gaussian has known nonzero mean; tests drift sign,
    # noise scale, preconditioning and Metropolis ratio together.
    loc = torch.zeros((1, 2), device=device)
    chol = torch.tensor([[[.7, 0.], [.2, 1.]]], device=device)
    direction = torch.tensor([.6, -.4], device=device)
    tilt = lambda z, h: -(z * direction).sum(-1)
    samples, acceptance = langevin(tilt, loc, loc, chol, 100, 8192, .25, 654)
    expected_mean = (chol @ chol.transpose(1, 2) @ direction[:, None]).squeeze(-1)
    torch.testing.assert_close(samples.mean(1), expected_mean, atol=.045, rtol=0)
    torch.testing.assert_close(torch.cov(samples[0].T), (chol @ chol.transpose(1, 2))[0], atol=.06, rtol=0)
    assert acceptance > .5
    forbidden = []
    for name in ['test.csv', 'source.csv', 'source.pt', 'scores.npy']:
        try:
            (root / name).open('rb')
        except RuntimeError:
            forbidden.append(name)
    assert len(forbidden) == 4
    save(root / f'selftest_{device}.json', dict(status='PASS', exact_joint_gaussian=True,
        exact_gaussian_gradient=True, tilted_gaussian_sampler=True, common_initialization=True,
        zero_initial_mc_mean=True, no_list_fallback=True, gradients_all_active_modules=True,
        forbidden=forbidden))
    print('AMBIGUITY_SELFTEST_PASS', device, flush=True)


class GroupedRatings:
    def __init__(self, data, overlap, device):
        active = data[data[:, 0] < overlap]
        active = active[np.argsort(active[:, 0], kind='stable')]
        users, starts, counts = np.unique(active[:, 0].astype(np.int64), return_index=True, return_counts=True)
        self.users = torch.tensor(users, device=device)
        self.starts = torch.tensor(starts, device=device)
        self.counts = torch.tensor(counts, device=device)
        self.items = torch.tensor(active[:, 1], device=device, dtype=torch.long)
        self.ratings = torch.tensor(active[:, 2], device=device, dtype=torch.float32)
        self.rows = len(active)

    def batch(self, idx, base):
        length = int(self.counts[idx].max())
        offset = torch.arange(length, device=idx.device)
        mask = offset[None, :] < self.counts[idx, None]
        positions = (self.starts[idx, None] + offset).clamp_max(len(self.items) - 1)
        items = self.items[positions]
        residual = (self.ratings[positions] - base[items]) * mask
        return self.users[idx], items, residual, mask.float()


def correction(model, features, seen, seed):
    model.eval()
    unique, inverse = torch.unique(features[:, 0], sorted=True, return_inverse=True)
    z = []
    for index, batch in enumerate(unique.split(model.cfg['eval_user_batch'])):
        z.append(model.user_mean(batch, seed + 100000 + index).detach())
    means = torch.cat(z)
    with torch.no_grad():
        values = (means[inverse] * model.target_items(features[:, 1])).sum(-1)
        values = torch.where(seen[features[:, 1]], values, 0.)
    return values.cpu().numpy()


def train(root, variant, beta, seed):
    check_code(root)
    student_guard(root)
    cfg = read(root / 'protocol.json')
    cell = root / f'beta_{beta}'
    manifest = read(cell / 'manifest.json')
    for path, expected in manifest['hashes'].items():
        if path != 'data/test.csv':
            assert digest(cell / path) == expected
    out = root / variant / f'beta_{beta}/seed_{seed}'
    assert not (out / 'started.json').exists()
    save(out / 'started.json', dict(job=os.environ['SLURM_JOB_ID'], gpu=os.environ['CUDA_VISIBLE_DEVICES'], time=time.time()))
    query = PRIOR / TASK / f'seed_{seed}/queries/DNN'
    record = read(query / 'complete.json')
    assert digest(query / 'topk_ids.npz') == record['item_ids_sha256']
    api = np.load(query / 'topk_ids.npz')
    np.testing.assert_array_equal(api['user_ids'], np.arange(manifest['overlap_users']))
    ids = api['item_ids']
    vocab = np.unique(ids)
    lists = torch.zeros((manifest['target_users'], cfg['k']), device='cuda', dtype=torch.long)
    lists[:len(ids)] = torch.tensor(np.searchsorted(vocab, ids) + 1, device='cuda')
    validation = frame(cell / 'data/validation.csv')
    vx = torch.tensor(validation[:, :2], device='cuda', dtype=torch.long)
    tx = torch.tensor(np.load(cell / 'data/test_features.npy'), device='cuda', dtype=torch.long)
    parent_record = read(PARENT / f'beta_{beta}/seed_{seed}/residual_real/complete.json')
    selection = None
    initial_hashes = []
    for phase in ['select', 'refit']:
        data = frame(cell / ('data/inner_train.csv' if phase == 'select' else 'data/train.csv'))
        test_users = np.load(cell / 'data/test_users.npy')
        assert not np.intersect1d(data[:, 0], test_users).size
        if phase == 'select':
            assert not np.intersect1d(data[:, 0], validation[:, 0]).size
        basepath = cell / ('inner_baseline' if phase == 'select' else 'baseline')
        base64 = predict(dict(np.load(basepath / 'statistics.npz')), np.arange(manifest['target_items']), 'item_mean')
        base = torch.tensor(base64.astype(np.float32), device='cuda')
        seen = torch.tensor(np.load(basepath / 'seen.npy'), device='cuda')
        seed_all(seed)
        model = ConditionalList(lists, len(vocab), manifest['target_items'], base, cfg, variant).cuda()
        initial_hashes.append(parameter_hash(model))
        common_hash = parameter_hash(model, True)
        assert common_hash == parent_record['initial_parameters_sha256']
        grouped = GroupedRatings(data, manifest['overlap_users'], 'cuda')
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(seed + 200)
        seed_all(seed + 100)
        if phase == 'select':
            baseval = base64[validation[:, 1].astype(int)]
            selection = dict(epoch=0, lambda_value=0., validation_rmse=float(np.sqrt(np.square(baseval - validation[:, 2]).mean())))
            torch.save(model.state_dict(), out / 'selected.pt')
            np.save(out / 'validation.npy', baseval)
        history = []
        epochs = cfg['epochs'] if phase == 'select' else selection['epoch']
        for epoch in range(1, epochs + 1):
            tick = time.monotonic()
            model.train()
            total = 0.
            accepts = []
            for batch_id, idx in enumerate(torch.randperm(len(grouped.users), device='cuda', generator=generator).split(cfg['user_batch_size'])):
                batch = grouped.batch(idx, base)
                opt.zero_grad(set_to_none=True)
                loss, extra = model.objective(*batch, seed + epoch * 10000 + batch_id)
                assert torch.isfinite(loss), (variant, epoch, batch_id)
                loss.backward()
                opt.step()
                count = int(batch[-1].sum())
                total += float(loss.detach()) * count
                if extra:
                    accepts.append([float(extra['prior_accept']), float(extra['posterior_accept'])])
            row = dict(epoch=epoch, objective_surrogate=total / grouped.rows, seconds=time.monotonic() - tick)
            if accepts:
                row['mala_acceptance'] = np.mean(accepts, axis=0).tolist()
            if phase == 'select':
                residual = correction(model, vx, seen, seed)
                row['validation'] = {}
                for lam in cfg['lambdas']:
                    pred = baseval + lam * residual
                    value = float(np.sqrt(np.square(pred - validation[:, 2]).mean()))
                    row['validation'][str(lam)] = value
                    if value < selection['validation_rmse']:
                        selection = dict(epoch=epoch, lambda_value=lam, validation_rmse=value)
                        torch.save(model.state_dict(), out / 'selected.pt')
                        np.save(out / 'validation.npy', pred)
            history.append(row)
            save(out / f'{phase}_history.json', history)
            print('AMBIGUITY_EPOCH', variant, beta, seed, phase, json.dumps(row), flush=True)
        save(out / f'{phase}_history.json', history)
        if phase == 'select':
            model.load_state_dict(torch.load(out / 'selected.pt', map_location='cuda', weights_only=True))
            np.testing.assert_allclose(baseval + selection['lambda_value'] * correction(model, vx, seen, seed), np.load(out / 'validation.npy'), atol=1e-7, rtol=0)
            save(out / 'selection.json', selection)
        else:
            assert initial_hashes[0] == initial_hashes[1]
            torch.save(model.state_dict(), out / 'final.pt')
            result = base64[tx[:, 1].cpu().numpy()] + selection['lambda_value'] * correction(model, tx, seen, seed)
            np.save(out / 'test.npy', result)
            model.load_state_dict(torch.load(out / 'final.pt', map_location='cuda', weights_only=True))
            np.testing.assert_array_equal(result, base64[tx[:, 1].cpu().numpy()] + selection['lambda_value'] * correction(model, tx, seen, seed))
            save(out / 'complete.json', dict(status='PASS', variant=variant, beta=beta, seed=seed,
                initial_parameters_sha256=initial_hashes[0], common_initial_parameters_sha256=common_hash,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                full_outer_train_rows=len(data), overlap_train_rows=grouped.rows, grouped_users=len(grouped.users),
                refit_epochs=epochs, refit_reinitialized=True, restored_predictions=True, test_labels_opened=False,
                source_ids_sha256=record['item_ids_sha256'], final_sha256=digest(out / 'final.pt'),
                selection_sha256=digest(out / 'selection.json'), test_sha256=digest(out / 'test.npy'), finished=time.time()))
        del model, grouped, opt, data
    print('AMBIGUITY_FROZEN', variant, beta, seed, flush=True)


def worker(root, index):
    for task in read(root / 'tasks.json'):
        out = root / task['variant'] / f"beta_{task['beta']}/seed_{task['seed']}"
        try:
            with (out / 'claim.json').open('x') as handle:
                json.dump(dict(worker=index, gpu=os.environ['CUDA_VISIBLE_DEVICES']), handle)
        except FileExistsError:
            continue
        command = [sys.executable, '-u', str(Path(__file__)), 'train', '--root', str(root),
                   '--variant', task['variant'], '--beta', str(task['beta']), '--seed', str(task['seed'])]
        print('WORKER_TASK', index, task, flush=True)
        with (out / 'train.log').open('w') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def metric(pred, y):
    return dict(mae=float(np.abs(pred - y).mean()), rmse=float(np.sqrt(np.square(pred - y).mean())))


def evaluate(root):
    check_code(root)
    cfg = read(root / 'protocol.json')
    assert read(root / 'selftest_cuda.json')['status'] == 'PASS'
    frozen = {}
    for task in read(root / 'tasks.json'):
        out = root / task['variant'] / f"beta_{task['beta']}/seed_{task['seed']}"
        done = read(out / 'complete.json')
        assert done['status'] == 'PASS' and not done['test_labels_opened']
        for filename, key in [('test.npy', 'test_sha256'), ('final.pt', 'final_sha256'), ('selection.json', 'selection_sha256')]:
            assert digest(out / filename) == done[key]
        frozen[str(out.relative_to(root))] = digest(out / 'complete.json')
    assert len(frozen) == 30
    with (root / 'evaluation_started.json').open('x') as handle:
        json.dump(frozen, handle, indent=2)
    rows, per_seed = [], []
    for beta in BETAS:
        cell = root / f'beta_{beta}'
        manifest = read(cell / 'manifest.json')
        assert digest(cell / 'data/test.csv') == manifest['hashes']['data/test.csv']
        test = frame(cell / 'data/test.csv')
        y = test[:, 2]
        np.testing.assert_array_equal(test[:, :2], np.load(cell / 'data/test_features.npy'))
        base = predict(dict(np.load(cell / 'baseline/statistics.npz')), test[:, 1].astype(int), 'item_mean')
        refs = {seed: metric(np.load(PARENT / f'beta_{beta}/seed_{seed}/residual_real/test.npy'), y) for seed in SEEDS}
        matched = {seed: metric(np.load(root / f'point_matched/beta_{beta}/seed_{seed}/test.npy'), y) for seed in SEEDS}
        for variant in ['item_mean', 'LIST_archived', *VARIANTS]:
            values = []
            for seed in SEEDS:
                if variant == 'item_mean':
                    scores, selection = metric(base, y), dict(epoch=0, lambda_value=0.)
                elif variant == 'LIST_archived':
                    scores = refs[seed]
                    selection = read(PARENT / f'beta_{beta}/seed_{seed}/residual_real/selection.json')
                else:
                    out = root / variant / f'beta_{beta}/seed_{seed}'
                    scores = metric(np.load(out / 'test.npy'), y)
                    selection = read(out / 'selection.json')
                row = dict(beta=beta, variant=variant, seed=seed, epoch=selection['epoch'], lambda_value=selection['lambda_value'], **scores)
                for name in ['mae', 'rmse']:
                    row['delta_' + name + '_vs_archived'] = scores[name] - refs[seed][name]
                    row['delta_' + name + '_vs_matched'] = scores[name] - matched[seed][name]
                values.append(row)
                per_seed.append(row)
            aggregate = dict(beta=beta, variant=variant)
            for key in ['mae', 'rmse', 'delta_mae_vs_archived', 'delta_rmse_vs_archived', 'delta_mae_vs_matched', 'delta_rmse_vs_matched']:
                aggregate[key] = float(np.mean([v[key] for v in values]))
            for key in ['mae', 'rmse']:
                aggregate[key + '_std'] = float(np.std([v[key] for v in values], ddof=1))
            rows.append(aggregate)
    for name, data in [('comparison', rows), ('per_seed', per_seed)]:
        with (root / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    save(root / 'summary.json', dict(status='PASS', comparison=rows, per_seed=per_seed, protocol=cfg))
    labels = {'item_mean': '普通物品均值', 'LIST_archived': '原版 LIST（已归档）',
              'point_matched': 'LIST（按用户成组训练对照）', 'gaussian_fixed': '条件高斯，固定方差',
              'gaussian_learned': '条件高斯，学习方差', 'energy_20': '条件能量 + 20步朗之万',
              'energy_60': '条件能量 + 60步朗之万'}
    lines = ['# LIST 跨域迁移歧义实验结果', '',
             '任务：Music→Movie 去重版，17,904名重叠用户。β为测试冷用户比例；每项为2020/2021/2022三个种子各自指标的均值±样本标准差，不是预测集成。', '',
             '| β | 方法 | MAE ± SD | RMSE ± SD | ΔMAE 对原版 | ΔRMSE 对原版 | ΔRMSE 对成组对照 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['beta']}% | {labels[row['variant']]} | {row['mae']:.6f} ± {row['mae_std']:.6f} | {row['rmse']:.6f} ± {row['rmse_std']:.6f} | {row['delta_mae_vs_archived']:+.6f} | {row['delta_rmse_vs_archived']:+.6f} | {row['delta_rmse_vs_matched']:+.6f} |")
    lines += ['', '差值为新版本减对照，负值更好。所有新版本使用同一64用户成组训练，保留每个用户全部目标训练评分；原版按2048交互训练。成组对照用于区分这项必要训练组织变化和概率模型变化。',
              '源DNN Top100、排名池化、32维表示、目标内积、普通均值、10轮上限、Adam与验证/refit规则一致。仅目标域独有用户的零残差常数项从新版本更新中省略，它们仍完整参与普通均值。',
              '固定高斯std=.3；学习方差范围std=.05..2；条件能量是高斯参考上的有界MLP修正。训练根据先验/训练后验MCMC估计联合似然梯度；推断只根据源列表采样，禁止使用真实待预测评分。',
              '朗之万使用Metropolis校正、32条训练链、128条预测链和配对随机数；20/60为训练及预测链长度。有限链仍是近似推断。',
              '全部30个最终模型和预测冻结后才读取测试标签。复用历史测试划分，结果属于探索性比较。此轮只比较推荐误差，没有增加隐私、机制或因果收益主张。', '',
              '[逐种子结果](per_seed.csv) · [汇总CSV](comparison.csv) · [完整协议](protocol.json) · [验证记录](verification.json)']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print('AMBIGUITY_RESULTS', json.dumps(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'selftest', 'train', 'worker', 'evaluate'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--beta', type=int, choices=BETAS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'train':
        train(root, args.variant, args.beta, args.seed)
    elif args.stage == 'selftest':
        selftest(root, args.device)
    elif args.stage == 'worker':
        worker(root, args.worker)
    else:
        globals()[args.stage](root)
