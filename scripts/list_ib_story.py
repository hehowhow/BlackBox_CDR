"""One Gaussian hidden bottleneck for frozen-protocol residual LIST.

Train subprocesses cannot open test labels, source histories, teacher weights or
teacher scores. All 18 predictions are frozen before the evaluate subprocess.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from small_task_comparison import PRIOR, TASK, SEEDS, BETAS, ResidualList, context, frame, corrections
from book_movie_protocol import BASE, digest, save, predict
from residual_experiment import seed_all
from list_source_backbones import student_guard, check_code, read
from residual_summary import metrics

PARENT = BASE/'runs/small_task_comparison_8650'
VARIANTS = {'noise_only': 0., 'ib_1e4': 1e-4, 'ib_1e3': 1e-3}


class IBResidual(ResidualList):
    """Gaussian before the existing zero output layer: exact zero warm start."""
    def __init__(self, lists, vocab, items, base, cfg):
        super().__init__(lists, vocab, items, base, cfg)
        self.logvar = nn.Linear(cfg['hidden'], cfg['hidden'])
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, cfg['initial_logvar'])
        self.bounds = cfg['logvar_bounds']

    def posterior(self, users):
        rows = self.lists[users]
        pooled = self.source_items(rows, per_sample_weights=self.rank_weights.expand_as(rows))
        mu = self.encoder[1](self.encoder[0](pooled))
        logvar = self.logvar(mu).clamp(*self.bounds)
        active = rows[:, 0] != 0
        return mu, logvar, active

    def correction_kl(self, users, items):
        mu, logvar, active = self.posterior(users)
        hidden = mu + torch.exp(.5*logvar)*torch.randn_like(mu) if self.training else mu
        z = self.encoder[3](self.encoder[2](hidden))*active[:, None]
        corr = (z*self.target_items(items)).sum(-1)
        # Diagonal-Gaussian KL sums over all 64 hidden dimensions, then the
        # caller averages over interaction rows. No-list rows contribute zero.
        kl = .5*(mu.square()+logvar.exp()-1-logvar).sum(-1)*active
        return corr, kl

    def correction(self, users, items):
        return self.correction_kl(users, items)[0]


def parameter_hash(model, common=False):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if not common or not name.startswith('logvar.'):
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    previous = read(PARENT/'protocol.json')
    assert read(PARENT/'verification.json')['status'] == 'PASS'
    for path, h in previous['code_hashes'].items():
        assert digest(Path(path)) == h, path
    paths = [Path(__file__), BASE/'scripts/slurm_list_ib_story.sh', BASE/'scripts/verify_list_ib_story.py']
    cfg = dict(previous, parent=str(PARENT), parent_protocol_sha256=digest(PARENT/'protocol.json'),
        variants=VARIANTS, arms=['real'], initial_logvar=-6., logvar_bounds=[-12., 6.],
        architecture='unchanged rank pool32 -> original Linear32/64+ReLU as Gaussian mean; new logvar Linear64/64; sample64 -> original dropout0.1 -> original zero Linear64/32 -> target dot',
        objective='interaction-mean rating MSE + eta * interaction-mean KL; KL summed over64 dimensions; no-list rows zero; repeated users weighted by rating count',
        inference='deterministic Gaussian mean; no MC sampling; equivalent original architecture at inference',
        fair_control='three eta values share all initialization, batch permutations and stochastic schedules; original common tensors exactly match8650',
        parameter_change='4160 additional trainable log-variance parameters, used only in training',
        evaluation='all18 frozen before first new evaluation opens test labels; reused historical test, exploratory',
        code_hashes={**previous['code_hashes'], **{str(p.resolve()):digest(p) for p in paths}})
    save(root/'protocol.json', cfg)
    for variant in VARIANTS:
        branch = root/variant
        branch.mkdir()
        save(branch/'protocol.json', dict(cfg, variant=variant, eta=VARIANTS[variant]))
        for beta in BETAS:
            cell = branch/f'beta_{beta}'
            cell.mkdir()
            old = PARENT/f'beta_{beta}'
            manifest = read(old/'manifest.json')
            # Test labels are not even hashed by preparation; evaluation audits
            # the frozen parent hash after every candidate has completed.
            for path, h in manifest['hashes'].items():
                if path != 'data/test.csv':
                    assert digest(old/path) == h, path
            for name in ['data', 'baseline', 'inner_baseline']:
                (cell/name).symlink_to((old/name).resolve(), target_is_directory=True)
            save(cell/'manifest.json', manifest)
            for seed in SEEDS:
                (cell/f'seed_{seed}/residual_real').mkdir(parents=True)
    save(root/'data_verification.json', dict(status='PASS',parent=8650,same_splits=True,
        no_test_labels_opened=True,variants_fixed_before_training=True))
    print('IB_PREPARED', root, flush=True)


def selftest(root):
    cfg = read(root/'protocol.json')
    check_code(root)
    student_guard(root)
    lists = torch.tensor([list(range(1,101)), list(range(100,0,-1)), [0]*100], device='cuda')
    base = torch.tensor([3.,4.,2.],device='cuda')
    users = torch.arange(3,device='cuda')
    seed_all(123)
    original = ResidualList(lists,100,3,base,cfg).cuda()
    seed_all(123)
    model = IBResidual(lists,100,3,base,cfg).cuda()
    assert parameter_hash(model,True) == parameter_hash(original)
    added = sum(p.numel() for p in model.parameters())-sum(p.numel() for p in original.parameters())
    assert added == 4160
    model.train()
    for _ in range(3):
        corr, kl = model.correction_kl(users,users)
        torch.testing.assert_close(corr,torch.zeros_like(corr),rtol=0,atol=0)
        assert kl[-1] == 0 and torch.isfinite(kl).all()
    mu, lv, active = model.posterior(users)
    manual = .5*(mu.square()+lv.exp()-1-lv).sum(-1)*active
    torch.testing.assert_close(kl,manual)
    model.eval()
    torch.testing.assert_close(model(users,users),original(users,users),rtol=0,atol=0)
    model.train()
    corr, kl = model.correction_kl(users,users)
    ((base+corr-5).square().mean()+1e-3*kl.mean()).backward()
    assert model.encoder[-1].weight.grad.abs().sum() > 0
    assert model.logvar.bias.grad.abs().sum() > 0
    with torch.no_grad():
        model.encoder[-1].weight.normal_(std=.1)
        model.encoder[-1].bias.fill_(1.)
        original.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('logvar.')})
    model.eval(); original.eval()
    torch.testing.assert_close(model(users,users),original(users,users),rtol=0,atol=0)
    assert model(users,users)[-1] == base[-1]
    model.train(); model.zero_grad()
    corr, _ = model.correction_kl(users,users)
    (base+corr-5).square().mean().backward()
    assert model.logvar.weight.grad.abs().sum() > 0
    assert model.source_items.weight.grad.abs().sum() > 0
    assert model.target_items.weight.grad.abs().sum() > 0
    forbidden = []
    for filename in ['test.csv','scores.npy','pretrained.pt','source.csv']:
        try:
            (root/filename).open('rb')
        except RuntimeError:
            forbidden.append(filename)
    assert len(forbidden) == 4
    save(root/'selftest.json',dict(status='PASS',zero_correction_during_sampling=True,
        exact_common_initialization=True,extra_parameters=added,gaussian_kl_formula=True,
        no_list_zero_kl_and_correction=True,inference_equivalent_original_mlp=True,
        gradient_to_variance_from_mse_and_kl=True,guard_blocks=forbidden))
    print('IB_SELFTEST_PASS',flush=True)


def train(root,variant,beta,seed):
    branch = root/variant
    check_code(branch); student_guard(branch)
    cell,cfg,manifest = context(branch,beta)
    out = cell/f'seed_{seed}/residual_real'
    assert not (out/'started.json').exists()
    save(out/'started.json',dict(job=os.environ['SLURM_JOB_ID'],gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    query = PRIOR/TASK/f'seed_{seed}/queries/DNN'
    query_record = read(query/'complete.json')
    assert digest(query/'topk_ids.npz') == query_record['item_ids_sha256']
    api = np.load(query/'topk_ids.npz')
    np.testing.assert_array_equal(api['user_ids'],np.arange(manifest['overlap_users']))
    ids = api['item_ids']; vocabulary = np.unique(ids)
    lists = torch.zeros((manifest['target_users'],cfg['k']),device='cuda',dtype=torch.long)
    lists[:len(ids)] = torch.tensor(np.searchsorted(vocabulary,ids)+1,device='cuda')
    np.save(out/'assignment.npy',np.arange(len(ids)))
    val = frame(cell/'data/validation.csv')
    vx = torch.tensor(val[:,:2],device='cuda',dtype=torch.long)
    tx = torch.tensor(np.load(cell/'data/test_features.npy'),device='cuda',dtype=torch.long)
    parent_record = read(PARENT/f'beta_{beta}/seed_{seed}/residual_real/complete.json')
    assert digest(out/'assignment.npy') == parent_record['assignment_sha256']
    selection = None; initial_hashes = []
    for phase in ['select','refit']:
        data = frame(cell/('data/inner_train.csv' if phase=='select' else 'data/train.csv'))
        basepath = cell/('inner_baseline' if phase=='select' else 'baseline')
        stats = dict(np.load(basepath/'statistics.npz'))
        base64 = predict(stats,np.arange(manifest['target_items']),'item_mean')
        base = torch.tensor(base64.astype(np.float32),device='cuda')
        seen = torch.tensor(np.load(basepath/'seen.npy'),device='cuda')
        seed_all(seed)
        model = IBResidual(lists,len(vocabulary),manifest['target_items'],base,cfg).cuda()
        initial_hashes.append(parameter_hash(model))
        common_hash = parameter_hash(model,True)
        assert common_hash == parent_record['initial_parameters_sha256']
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        seed_all(seed+100)
        x = torch.tensor(data[:,:2],device='cuda',dtype=torch.long)
        y = torch.tensor(data[:,2],device='cuda',dtype=torch.float32)
        active_rows = int((data[:,0]<manifest['overlap_users']).sum())
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(seed+200)
        if phase == 'select':
            np.testing.assert_array_equal(corrections(model,vx,seen),np.zeros(len(vx),dtype=np.float32))
            baseval = base64[val[:,1].astype(int)]
            selection = dict(epoch=0,lambda_value=0.,validation_rmse=float(np.sqrt(np.square(baseval-val[:,2]).mean())))
            torch.save(model.state_dict(),out/'selected.pt');np.save(out/'validation.npy',baseval)
        epochs = cfg['epochs'] if phase=='select' else selection['epoch']
        history = []
        for epoch in range(1,epochs+1):
            started = time.monotonic();model.train()
            totals = torch.zeros(3,device='cuda')
            for ix in torch.randperm(len(x),device='cuda',generator=generator).split(cfg['batch_size']):
                opt.zero_grad(set_to_none=True)
                corr, kl = model.correction_kl(x[ix,0],x[ix,1])
                mse = (base[x[ix,1]]+corr-y[ix]).square().mean()
                kl_mean = kl.mean();loss = mse+cfg['eta']*kl_mean
                loss.backward();opt.step()
                totals += torch.stack([mse.detach(),kl_mean.detach(),loss.detach()])*len(ix)
            values = (totals/len(x)).cpu().tolist()
            row = dict(epoch=epoch,train_mse=values[0],kl_per_all_interactions=values[1],
                kl_per_source_list_interaction=values[1]*len(x)/active_rows,objective=values[2],
                seconds=time.monotonic()-started)
            assert np.isfinite(values).all()
            if phase == 'select':
                correction = corrections(model,vx,seen);row['validation'] = {}
                for lam in cfg['lambdas']:
                    pred = baseval+lam*correction
                    value = float(np.sqrt(np.square(pred-val[:,2]).mean()))
                    row['validation'][str(lam)] = value
                    if value < selection['validation_rmse']:
                        selection = dict(epoch=epoch,lambda_value=lam,validation_rmse=value)
                        torch.save(model.state_dict(),out/'selected.pt');np.save(out/'validation.npy',pred)
            history.append(row);save(out/f'{phase}_history.json',history)
            print('IB_EPOCH',variant,beta,seed,phase,json.dumps(row),flush=True)
        if phase == 'select':
            model.load_state_dict(torch.load(out/'selected.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(baseval+selection['lambda_value']*corrections(model,vx,seen),np.load(out/'validation.npy'))
            save(out/'selection.json',selection)
        else:
            assert initial_hashes[0] == initial_hashes[1]
            torch.save(model.state_dict(),out/'final.pt')
            p = base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(model,tx,seen)
            np.save(out/'test.npy',p);save(out/'refit_history.json',history)
            restored = IBResidual(lists,len(vocabulary),manifest['target_items'],base,cfg).cuda()
            restored.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(base64[tx[:,1].cpu().numpy()]+selection['lambda_value']*corrections(restored,tx,seen),p)
            save(out/'complete.json',dict(status='PASS',beta=beta,variant=variant,seed=seed,eta=cfg['eta'],
                selection_sha256=digest(out/'selection.json'),final_sha256=digest(out/'final.pt'),
                test_sha256=digest(out/'test.npy'),assignment_sha256=digest(out/'assignment.npy'),
                initial_parameters_sha256=initial_hashes[0],shared_initial_parameters_sha256=common_hash,
                trainable_parameters=trainable,extra_parameters=4160,full_outer_train_rows=len(data),
                source_list_interactions=active_rows,refit_epochs=epochs,refit_reinitialized=True,
                restored_predictions=True,test_labels_opened=False,source_ids_sha256=query_record['item_ids_sha256']))
            del restored
        del model,opt,x,y,data
    print('IB_FROZEN',variant,beta,seed,flush=True)


def evaluate(root):
    check_code(root);cfg = read(root/'protocol.json')
    assert read(root/'selftest.json')['status'] == 'PASS'
    frozen = {}
    for variant in VARIANTS:
        for beta in BETAS:
            for seed in SEEDS:
                out = root/variant/f'beta_{beta}/seed_{seed}/residual_real'
                done = read(out/'complete.json')
                assert done['status']=='PASS' and done['test_labels_opened'] is False
                for name,key in [('test.npy','test_sha256'),('final.pt','final_sha256'),('selection.json','selection_sha256')]:
                    assert digest(out/name)==done[key]
                peer=read(root/'noise_only'/f'beta_{beta}/seed_{seed}/residual_real/complete.json')
                assert done['initial_parameters_sha256']==peer['initial_parameters_sha256']
                frozen[str(out.relative_to(root))]=digest(out/'complete.json')
    assert len(frozen)==18
    with (root/'evaluation_started.json').open('x') as f:json.dump(frozen,f,indent=2)
    rows=[];seed_rows=[];results={}
    for beta in BETAS:
        cell=PARENT/f'beta_{beta}';manifest=read(cell/'manifest.json')
        assert digest(cell/'data/test.csv')==manifest['hashes']['data/test.csv']
        test=frame(cell/'data/test.csv');y=test[:,2];_,users=np.unique(test[:,0],return_inverse=True)
        np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
        baseline=predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean')
        parent_predictions={s:np.load(cell/f'seed_{s}/residual_real/test.npy') for s in SEEDS}
        parent_metrics={s:metrics(parent_predictions[s],y,users) for s in SEEDS}
        baseline_metrics=metrics(baseline,y,users)
        results[str(beta)]={}
        for variant in ['item_mean','LIST_original',*VARIANTS]:
            individual=[]
            for seed in SEEDS:
                if variant=='item_mean':
                    values=baseline_metrics;selected={'epoch':0,'lambda_value':0.}
                elif variant=='LIST_original':
                    values=parent_metrics[seed];selected=read(cell/f'seed_{seed}/residual_real/selection.json')
                else:
                    out=root/variant/f'beta_{beta}/seed_{seed}/residual_real'
                    values=metrics(np.load(out/'test.npy'),y,users);selected=read(out/'selection.json')
                row=dict(beta=beta,variant=variant,seed=seed,eta=VARIANTS.get(variant,''),
                    epoch=selected['epoch'],lambda_value=selected['lambda_value'],mae=values['mae'],rmse=values['rmse'],
                    delta_mae_vs_original=values['mae']-parent_metrics[seed]['mae'],
                    delta_rmse_vs_original=values['rmse']-parent_metrics[seed]['rmse'])
                individual.append(row);seed_rows.append(row)
            aggregate=dict(beta=beta,variant=variant,seed='mean_of_2020_2021_2022',eta=VARIANTS.get(variant,''))
            for key in ['mae','rmse','delta_mae_vs_original','delta_rmse_vs_original']:
                aggregate[key]=float(np.mean([v[key] for v in individual]))
            for key in ['mae','rmse']:
                aggregate[key+'_std']=float(np.std([v[key] for v in individual],ddof=1))
            rows.append(aggregate)
            results[str(beta)][variant]=dict(aggregate=aggregate,per_seed=individual)
    save(root/'summary.json',dict(status='PASS',results=results,protocol=cfg,
        parent_summary_sha256=digest(PARENT/'summary.json'),scope='exploratory reused split; no robustness or explanation claim'))
    for name,data in [('comparison',rows),('per_seed',seed_rows)]:
        with (root/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save(root/'verification.json',dict(status='PASS',all18_frozen_before_test=True,
        same_common_initialization_as8650=True,identical_initialization_across_eta=True,
        independent_verifier_pending=True,metric_rows=seed_rows))
    lines=['# LIST IB 叙事线实验','',
        '固定 Music→Movie 去重版、DNN Top100 ID/rank、β20/80、seeds2020/2021/2022。普通物品均值，无平滑。',
        '原排名聚合32→原Linear32/64+ReLU作为高斯均值；新logvar头64→64；训练采样后接原Dropout0.1及原零初始化Linear64/32，再与目标物品点积。推断取高斯均值，因而推断结构与原LIST一致。新增4160个训练参数。',
        '三个固定η=0/1e-4/1e-3；logvar初值-6、限制[-12,6]；KL按64维求和，只对有列表的交互计入，再按全部训练交互平均。目标域独有用户KL与修正均为0；同一用户按其交互次数加权。',
        '训练时λ=1；内层冷用户验证在epoch0..10与λ={0,.25,.5,.75,1}中选最小RMSE，从完全相同初始化在完整训练集重训。Adam.001、weight decay1e-5、batch2048、dropout.1。',
        '所有18个候选的模型及预测冻结后才读取测试标签。原LIST与普通均值引用8650；同一历史测试集已被多次使用，本轮为探索性结果。此实验检验预测效果，不直接证明泛化鲁棒性或解释性。','',
        '| β | 版本 | MAE ± SD | RMSE ± SD | ΔRMSE 对原LIST |','|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['beta']}% | {row['variant']} | {row['mae']:.6f} ± {row['mae_std']:.6f} | {row['rmse']:.6f} ± {row['rmse_std']:.6f} | {row['delta_rmse_vs_original']:+.6f} |")
    lines+=['','差值为新版本减原LIST，负值更好。η=0是同一随机结构的无KL对照；它与原LIST的差异包括训练噪声和新方差参数。η>0对η=0的比较才更直接检验KL的增量价值。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('IB_EVALUATION_PASS',json.dumps(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','selftest','train','evaluate'])
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--beta',type=int,choices=BETAS)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args();root=args.root.resolve()
    if args.stage=='train':train(root,args.variant,args.beta,args.seed)
    else:globals()[args.stage](root)
