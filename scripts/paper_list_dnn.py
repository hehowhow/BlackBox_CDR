"""Paper task directions/beta, cleaned catalogs, DNN Top100 LIST vs item mean.

Inner cold-user validation selects epochs; a fresh final fit restores ALL outer
training users. Target test labels are inaccessible until every fit is frozen.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from book_movie_protocol import BASE, digest, save, fit, predict
from residual_experiment import seed_all
from list_source_backbones import ListStudent, infer, read, student_guard, check_code, selftest
from list_target_heads import paired_seed_mean
from residual_summary import metrics

TASKS = {'book_movie': ('Books', 'Movies_and_TV', 37388),
         'book_music': ('Books', 'CDs_and_Vinyl', 16738),
         'music_movie_clean': ('CDs_and_Vinyl', 'Movies_and_TV', 18031)}
SEEDS = [2020, 2021, 2022]
BETAS = [20, 50, 80]


def frame(path):
    return pd.read_csv(path, header=None, names=['uid', 'iid', 'y'])


def baseline(folder, train, item_count):
    folder.mkdir()
    stats = fit(train)
    np.savez(folder/'statistics.npz', **stats)
    ids = np.arange(item_count)
    values = predict(stats, ids, 'item_mean')
    reference = pd.Series(ids).map(train.groupby('iid').y.mean()).fillna(train.y.mean()).to_numpy()
    np.testing.assert_allclose(values, reference, rtol=0, atol=1e-12)
    np.save(folder/'values.npy', values.astype(np.float32))
    np.save(folder/'seen.npy', np.isin(ids, stats['ids']))
    save(folder/'audit.json', dict(status='PASS', independent_groupby_matches=True,
                                  method='ordinary item mean; unseen items -> training global mean',
                                  training_rows=len(train), training_global_mean=float(stats['global_mean'])))


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    files = ['paper_list_dnn.py', 'slurm_paper_list_dnn.sh', 'list_source_transfer.py',
             'list_source_backbones.py', 'list_target_heads.py', 'pfcdr_backbone_study.py',
             'book_movie_protocol.py', 'residual_experiment.py', 'residual_summary.py', 'run_list_encoder.py']
    paths = [BASE/'scripts'/p for p in files]
    paths += [BASE/'external/PFCDR'/p for p in ['models.py', 'float_embedding.py', 'inversion.py', 'inversion_config.json']]
    cfg = dict(tasks=list(TASKS), seeds=SEEDS, betas=BETAS, seed=2020,
               source_dim=10, source_epochs=10, source_lr=.01, source_weight_decay=0., source_batch=512,
               source_batch_policy='retain best existing DNN LIST pipeline (512 for every task); not paper task3 batch256',
               source_backbone='author affine DNN; fixed final epoch; source only; shared across beta within task/seed',
               k=100, seen_item_filter=False, score_clipping=False, learner_uses_scores=False,
               dim=32, hidden=64, dropout=.1, epochs=10, lr=.001, weight_decay=1e-5, batch_size=2048,
               student='original LIST: rank-weight pooling, MLP encoder, dot target item + train mean + item bias',
               vocabulary='DNN returned opaque IDs only', baseline='item_mean only; no smoothing',
               split_seed=2020, validation_seed=20260923, inner_validation_fraction=.2,
               split='sorted raw IDs; nested test sets from one random permutation; round(beta*N), same count rule as paper',
               selection='inner cold-user validation RMSE, epoch1..10; fresh refit full outer training for selected epochs',
               test_policy='all 27 final models frozen and restored before target test labels are opened',
               scope='cleaned paper directions and all beta; three model seeds, one fixed split; not original PFCDR reproduction',
               cleanup='remove every cross-domain shared raw item from source only; target untouched; no renewed 5-core',
               bootstrap_repeats=2000, bootstrap_seed=997,
               code_hashes={str(p.resolve()):digest(p) for p in paths},
               input_hashes={n:digest(BASE/f'data/mid/{n}.csv') for n in ['Books','Movies_and_TV','CDs_and_Vinyl']})
    save(root/'protocol.json', cfg)
    for task, (sn, tn, expected) in TASKS.items():
        tr = root/task
        (tr/'source_data/id_maps').mkdir(parents=True)
        a, b = pd.read_csv(BASE/f'data/mid/{sn}.csv'), pd.read_csv(BASE/f'data/mid/{tn}.csv')
        su, tu = set(a.uid), set(b.uid)
        before = len(su & tu)
        assert before == expected, (task, before, expected)
        common = set(a.iid)&set(b.iid)
        old_rows = len(a)
        a = a[~a.iid.isin(common)].copy()
        co = sorted(set(a.uid)&tu)
        n = len(co)
        cleanup = dict(raw_overlap_users=before, clean_overlap_users=n, lost_overlap_users=before-n,
                       removed_items=len(common), removed_source_ratings=old_rows-len(a), target_ratings_unchanged=len(b))
        save(tr/'cleanup.json', cleanup)
        save(tr/'removed_source_items.json', sorted(common))
        maps = {}
        for label, vals in [('source_users',co+sorted(set(a.uid)-set(co))),
                            ('target_users',co+sorted(tu-set(co))),
                            ('source_items',sorted(set(a.iid))),('target_items',sorted(set(b.iid)))]:
            maps[label] = pd.DataFrame(dict(raw_id=vals, id=range(len(vals))))
            maps[label].to_csv(tr/f'source_data/id_maps/{label}.csv', index=False)
        assert not set(maps['source_items'].raw_id)&set(maps['target_items'].raw_id)
        assert not set(common)&set(maps['source_items'].raw_id)
        for f, prefix in [(a,'source'),(b,'target')]:
            for col, suffix in [('uid','users'),('iid','items')]:
                m = maps[f'{prefix}_{suffix}']
                f[col] = f[col].map(dict(zip(m.raw_id,m.id)))
            assert not f.isna().any().any()
            assert not f.duplicated(['uid','iid']).any()
            assert f.y.between(1,5).all()
        a.to_csv(tr/'source_data/source.csv', index=False, header=False)
        manifest = dict(task=task, source=sn, target=tn, overlap_users=n,
                        source_users=len(maps['source_users']), target_users=len(maps['target_users']),
                        source_items=len(maps['source_items']), target_items=len(maps['target_items']),
                        rows=dict(source=len(a)), cleanup=cleanup,
                        cross_domain_raw_item_overlap=0, cross_domain_same_user_item_rating_leaks=0,
                        hashes={'data/source.csv':digest(tr/'source_data/source.csv')})
        save(tr/'manifest.json', manifest)
        order = random.Random(cfg['split_seed']).sample(list(range(n)),n)
        for beta in BETAS:
            cell = tr/f'beta_{beta}'
            (cell/'data').mkdir(parents=True)
            test_ids = np.array(sorted(order[:round(beta*n/100)]),dtype=np.int64)
            outer_ids = np.setdiff1d(np.arange(n),test_ids)
            val_ids = np.array(sorted(random.Random(cfg['validation_seed']).sample(outer_ids.tolist(),round(.2*len(outer_ids)))),dtype=np.int64)
            inner_ids = np.setdiff1d(outer_ids,val_ids)
            outer=b[~b.uid.isin(test_ids)].copy()
            inner=outer[~outer.uid.isin(val_ids)].copy()
            val=b[b.uid.isin(val_ids)].copy()
            test=b[b.uid.isin(test_ids)].copy()
            assert len(outer)+len(test)==len(b)
            assert len(inner)+len(val)==len(outer)
            assert not set(outer.uid)&set(test.uid)
            assert not set(inner.uid)&set(val.uid)
            assert set(outer.uid)&set(range(n))==set(outer_ids)
            assert set(val.uid)==set(val_ids) and set(test.uid)==set(test_ids)
            for name, f in [('train',outer),('inner_train',inner),('validation',val),('test',test)]:
                f.to_csv(cell/f'data/{name}.csv', index=False, header=False)
            for name, ids in [('train',outer_ids),('inner_train',inner_ids),('validation',val_ids),('test',test_ids)]:
                np.save(cell/f'data/{name}_users.npy',ids)
            np.save(cell/'data/test_features.npy',test[['uid','iid']].to_numpy(dtype=np.int64))
            baseline(cell/'baseline',outer,manifest['target_items'])
            baseline(cell/'inner_baseline',inner,manifest['target_items'])
            cm=dict(manifest,beta=beta,users=dict(outer_train=len(outer_ids),inner_train=len(inner_ids),
                    validation=len(val_ids),test=len(test_ids),target_only=len(tu)-n),
                    rows=dict(source=len(a),train=len(outer),inner_train=len(inner),validation=len(val),test=len(test)),
                    hashes={str(p.relative_to(cell)):digest(p) for d in ['data','baseline','inner_baseline'] for p in sorted((cell/d).glob('*'))})
            save(cell/'manifest.json',cm)
            for seed in SEEDS:
                (cell/f'seed_{seed}').mkdir()
            print('PREPARED_CELL',task,beta,cm['users'],cm['rows'],flush=True)
        for seed in SEEDS:
            sub=tr/f'seed_{seed}'
            (sub/'queries/DNN').mkdir(parents=True)
            (sub/'data').symlink_to((tr/'source_data').resolve(),target_is_directory=True)
            save(sub/'manifest.json',manifest)
            save(sub/'protocol.json',dict(cfg,task=task,seed=seed))
        print('CLEANUP',task,json.dumps(cleanup),flush=True)
    save(root/'data_verification.json',dict(status='PASS',raw_counts_match_paper=True,all_catalog_intersections_zero=True,
                                         independent_item_means=True,disjoint_target_train_test_users=True))


def source(root,task,seed):
    from list_source_transfer import source as original_source
    original_source(root,task,seed,'DNN')


def train(root,task,seed,beta):
    student_guard(root)
    check_code(root)
    cfg=read(root/'protocol.json')
    cell=root/task/f'beta_{beta}'
    out=cell/f'seed_{seed}'
    manifest=read(cell/'manifest.json')
    assert not (out/'started.json').exists()
    save(out/'started.json',dict(job=os.environ['SLURM_JOB_ID'],gpu=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name()))
    for p,h in manifest['hashes'].items():
        if p!='data/test.csv': assert digest(cell/p)==h,p
    query=root/task/f'seed_{seed}/queries/DNN'
    assert digest(query/'topk_ids.npz')==read(query/'complete.json')['item_ids_sha256']
    api=np.load(query/'topk_ids.npz')
    np.testing.assert_array_equal(api['user_ids'],np.arange(manifest['overlap_users']))
    ids=api['item_ids']; vocab=np.unique(ids)
    compact=np.searchsorted(vocab,ids)
    np.testing.assert_array_equal(vocab[compact],ids)
    np.save(out/'vocabulary.npy',vocab)
    lists=torch.zeros((manifest['target_users'],cfg['k']),device='cuda',dtype=torch.long)
    lists[:len(ids)]=torch.tensor(compact+1,device='cuda')
    validation=frame(cell/'data/validation.csv').to_numpy()
    vx=torch.tensor(validation[:,:2],device='cuda',dtype=torch.long)
    tx=torch.tensor(np.load(cell/'data/test_features.npy'),device='cuda',dtype=torch.long)
    initial_hashes=[]
    selection=None
    for stage in ['select','refit']:
        tr=frame(cell/('data/inner_train.csv' if stage=='select' else 'data/train.csv')).to_numpy()
        train_users=np.unique(tr[:,0])
        assert not np.intersect1d(train_users,np.load(cell/'data/test_users.npy')).size
        if stage=='select': assert not np.intersect1d(train_users,validation[:,0]).size
        else: assert np.isin(np.load(cell/'data/validation_users.npy'),train_users).all()
        base=cell/('inner_baseline' if stage=='select' else 'baseline')
        seen=torch.tensor(np.load(base/'seen.npy'),device='cuda')
        fallback=torch.tensor(np.load(base/'values.npy'),device='cuda')
        seed_all(seed)
        mean=torch.tensor(tr[:,2],dtype=torch.float32).mean().item()
        model=ListStudent(lists,len(vocab),manifest['target_items'],mean,cfg).cuda()
        ph=hashlib.sha256()
        for p in model.parameters(): ph.update(p.detach().cpu().numpy().tobytes())
        initial_hashes.append(ph.hexdigest())
        seed_all(seed+100)
        opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        generator=torch.Generator(device='cuda').manual_seed(seed+200)
        x=torch.tensor(tr[:,:2],device='cuda',dtype=torch.long)
        y=torch.tensor(tr[:,2],device='cuda',dtype=torch.float32)
        epochs=cfg['epochs'] if stage=='select' else selection['epoch']
        history=[]
        for epoch in range(1,epochs+1):
            tick=time.monotonic(); model.train(); total=torch.zeros((),device='cuda')
            for idx in torch.randperm(len(x),generator=generator,device='cuda').split(cfg['batch_size']):
                opt.zero_grad(set_to_none=True)
                loss=(model(x[idx,0],x[idx,1])-y[idx]).square().mean()
                loss.backward(); opt.step(); total+=loss.detach()*len(idx)
            row=dict(epoch=epoch,train_mse=total.item()/len(x),seconds=time.monotonic()-tick)
            assert np.isfinite(row['train_mse'])
            if stage=='select':
                vp=infer(model,vx,seen,fallback)
                row.update(validation_rmse=float(np.sqrt(np.square(vp-validation[:,2]).mean())),validation_mae=float(np.abs(vp-validation[:,2]).mean()))
                if selection is None or row['validation_rmse']<selection['validation_rmse']:
                    selection=dict(row)
                    torch.save(model.state_dict(),out/'inner_selected.pt')
                    np.save(out/'validation.npy',vp)
            history.append(row); save(out/f'{stage}_history.json',history)
            print('LIST_EPOCH',task,beta,seed,stage,json.dumps(row),flush=True)
        if stage=='select':
            model.load_state_dict(torch.load(out/'inner_selected.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(infer(model,vx,seen,fallback),np.load(out/'validation.npy'))
            selection.update(validation_sha256=digest(out/'validation.npy'),checkpoint_sha256=digest(out/'inner_selected.pt'))
            save(out/'selection.json',selection)
        else:
            assert initial_hashes[0]==initial_hashes[1]
            torch.save(model.state_dict(),out/'final.pt')
            pred=infer(model,tx,seen,fallback)
            assert np.isfinite(pred).all()
            np.save(out/'test.npy',pred)
            # Independently reconstruct the original class and replay every prediction.
            restored=ListStudent(lists,len(vocab),manifest['target_items'],mean,cfg).cuda()
            restored.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True))
            np.testing.assert_array_equal(infer(restored,tx,seen,fallback),pred)
            save(out/'complete.json',dict(status='PASS',selected_epoch=selection['epoch'],refit_epochs=epochs,
                selection_sha256=digest(out/'selection.json'),final_sha256=digest(out/'final.pt'),
                test_sha256=digest(out/'test.npy'),list_sha256=digest(query/'topk_ids.npz'),
                initial_trainable_parameters_sha256=initial_hashes[0],refit_reinitialized=True,
                restored_predictions=True,full_outer_train_rows=len(tr),test_labels_opened=False))
        del model,opt,x,y
    print('FINAL_FROZEN',task,beta,seed,flush=True)


def evaluate(root):
    check_code(root)
    assert read(root/'selftest.json')['status']=='PASS'
    cfg=read(root/'protocol.json')
    frozen={}
    for task in TASKS:
        for beta in BETAS:
            for seed in SEEDS:
                p=root/task/f'beta_{beta}/seed_{seed}'
                c=read(p/'complete.json'); assert c['status']=='PASS'
                for filename,key in [('selection.json','selection_sha256'),('final.pt','final_sha256'),('test.npy','test_sha256')]:
                    assert digest(p/filename)==c[key]
                assert len(read(p/'refit_history.json'))==c['selected_epoch']==c['refit_epochs']
                assert c['selected_epoch']==min(read(p/'select_history.json'),key=lambda x:x['validation_rmse'])['epoch']
                frozen[str(p.relative_to(root))]=digest(p/'complete.json')
    with (root/'evaluation_started.json').open('x') as f: json.dump(frozen,f,indent=2)
    results={}; rows=[]; seed_rows=[]
    for task in TASKS:
        results[task]={}
        for beta in BETAS:
            cell=root/task/f'beta_{beta}'; m=read(cell/'manifest.json')
            for p,h in m['hashes'].items(): assert digest(cell/p)==h
            test=frame(cell/'data/test.csv').to_numpy()
            np.testing.assert_array_equal(test[:,:2],np.load(cell/'data/test_features.npy'))
            np.testing.assert_array_equal(np.unique(test[:,0]),np.load(cell/'data/test_users.npy'))
            y=test[:,2]; _,users=np.unique(test[:,0],return_inverse=True)
            mean=predict(dict(np.load(cell/'baseline/statistics.npz')),test[:,1].astype(int),'item_mean')
            # Independent mean recomputation from final train only (no source/test ratings).
            tr=frame(cell/'data/train.csv')
            ref=pd.Series(test[:,1].astype(int)).map(tr.groupby('iid').y.mean()).fillna(tr.y.mean()).to_numpy()
            np.testing.assert_allclose(mean,ref,rtol=0,atol=1e-12)
            predictions=np.stack([np.load(cell/f'seed_{s}/test.npy') for s in SEEDS])
            per_seed={str(s):metrics(predictions[i],y,users) for i,s in enumerate(SEEDS)}
            agg={k:dict(mean=float(np.mean([v[k] for v in per_seed.values()])),std=float(np.std([v[k] for v in per_seed.values()],ddof=1))) for k in ['mae','rmse']}
            comparison=paired_seed_mean(predictions,mean[None,:],y,users,cfg)
            comparison['interpretation']='negative favors LIST; user cluster 95% interval conditional on three fitted source/student seeds and one split; no multiplicity correction'
            bm=metrics(mean,y,users)
            result=dict(manifest=m,baseline=bm,aggregate=agg,per_seed=per_seed,comparison=comparison)
            results[task][str(beta)]=result
            save(cell/'results.json',dict(status='PASS',**result))
            rows.append(dict(task=task,beta=beta,overlap_users=m['overlap_users'],train_overlap_users=m['users']['outer_train'],
                test_users=m['users']['test'],mean_mae=bm['mae'],mean_rmse=bm['rmse'],
                list_mae=agg['mae']['mean'],list_mae_std=agg['mae']['std'],list_rmse=agg['rmse']['mean'],list_rmse_std=agg['rmse']['std'],
                delta_mae=comparison['delta_mae'],delta_rmse=comparison['delta_rmse'],
                delta_rmse_ci_low=comparison['rmse_95ci'][0],delta_rmse_ci_high=comparison['rmse_95ci'][1]))
            for s,v in per_seed.items(): seed_rows.append(dict(task=task,beta=beta,seed=s,epoch=read(cell/f'seed_{s}/selection.json')['epoch'],mae=v['mae'],rmse=v['rmse']))
    save(root/'summary.json',dict(status='PASS',protocol=cfg,results=results))
    save(root/'verification.json',dict(status='PASS',final_models=27,source_models=9,all_refits_frozen_before_test=True,
                                     all_prediction_restores_exact=True,independent_test_item_means_match=True))
    for name,data in [('comparison',rows),('per_seed',seed_rows)]:
        with (root/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    lines=['# 论文任务补齐：DNN 列表 LIST 与普通物品均值','',
        '三个论文方向 × β=20%/50%/80% × 三个源/学生种子；仅运行去重数据。数值越低越好。',
        'LIST 为三个种子指标均值；不是预测集成。普通均值仅使用同一最终训练集的物品评分，未见物品回退训练全局均值。','',
        '| 任务 | β | 重叠用户 | 最终训练重叠用户 | 测试用户 | 均值 MAE | LIST MAE | 均值 RMSE | LIST RMSE ± 标准差 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['task']} | {r['beta']}% | {r['overlap_users']} | {r['train_overlap_users']} | {r['test_users']} | {r['mean_mae']:.6f} | {r['list_mae']:.6f} | {r['mean_rmse']:.6f} | {r['list_rmse']:.6f} ± {r['list_rmse_std']:.6f} |")
    lines+=['','## 去重记录','','| 任务 | 论文重叠用户 | 清洗后 | 删除源物品 | 删除源评分 |','| --- | ---: | ---: | ---: | ---: |']
    for task in TASKS:
        c=read(root/task/'cleanup.json')
        lines.append(f"| {task} | {c['raw_overlap_users']} | {c['clean_overlap_users']} | {c['removed_items']} | {c['removed_source_ratings']} |")
    lines+=['','## 评价边界','',
        '保留全部目标独有训练用户。先从外层训练重叠用户中留出20%作冷用户验证，用10轮中最低验证RMSE选轮数，再重新初始化并用全部外层训练用户训练所选轮数。均值也使用全部外层训练数据。',
        '因此验证用户可以在最终重训时进入训练；测试用户始终不能进入目标训练。所有27个最终模型冻结并恢复预测核验后才统一评价测试标签。',
        '三个比例使用嵌套测试用户集合及一个固定划分，三个种子同时改变源模型和LIST。源数据与目标划分无关，同任务/种子的源模型在三个比例复用。',
        '删除源域所有与目标共享的原始物品ID，目标域不变，不重新执行5-core。各任务源目标物品交集均为零。Music→Movie清洗后的用户数不冒充论文人数。',
        '源模型沿用表现最好的既有DNN LIST参数：维度10、10轮、Adam 0.01、batch512；目标LIST维度32、hidden64、dropout0.1、Adam0.001、weight_decay1e-5、batch2048。DNN指作者仿射用户变换，目标评分仍是原LIST内积。',
        '对齐的是论文方向、完整原始数据和三档测试比例，不是论文PFCDR方法或其十次重复。特别是Music→Movie源batch沿用现有512，未改为作者该任务256。',
        '划分按排序原始ID和固定随机顺序生成，人数取round，与作者计数规则一致；不保证与作者未发布的用户名单相同，也不与此前60/20/20结果直接横比。',
        'LIST只接收源Top100的ID和排名，不接收源评分或权重。不裁剪分数，不过滤源已交互物品。没有平滑均值，也没有依据测试结果调参。',
        '置信区间见summary.json，按测试用户簇重采样，条件于已训练模型和固定划分；未做多重比较校正。历史原始数据已被探索，本轮不称全新独立盲测。','']
    (root/'report.md').write_text('\n'.join(lines))
    print('PAPER_LIST_ALL_PASS',json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','selftest','source','train','evaluate'])
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--task',choices=list(TASKS)); p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--beta',type=int,choices=BETAS)
    a=p.parse_args(); root=a.root.resolve()
    if a.stage=='source': source(root,a.task,a.seed)
    elif a.stage=='train': train(root,a.task,a.seed,a.beta)
    else: globals()[a.stage](root)
