"""Two independent fixed SID changes against original LIST on three tasks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from book_movie_protocol import BASE, digest, predict, save
from list_source_backbones import check_code, read, student_guard
from residual_experiment import seed_all
import small_task_comparison as archived
from small_task_comparison import ResidualList, corrections, frame

DATA_PARENT = BASE / "runs/paper_list_dnn_8638"
TASKS = {"music_movie_clean": 17904, "book_movie": 37388, "book_music": 16738}
BETAS, SEEDS = [20, 80], [2020, 2021, 2022]
VARIANTS = ["LIST_v1", "source_sid", "target_sid_aux"]
TOKENIZER_SEED = 20260927


def output(root, task, variant, beta, seed):
    if variant == "LIST_v1":
        return root / task / f"beta_{beta}/seed_{seed}/residual_real"
    return root / task / variant / f"beta_{beta}/seed_{seed}"


def old_baseline(task, beta, seed):
    parent = (BASE / "runs/small_task_comparison_8650" if task == "music_movie_clean"
              else BASE / "runs/list_ambiguity_transfer_8923" / task)
    return parent / f"beta_{beta}/seed_{seed}/residual_real"


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare(root):
    from list_sid_source_model import SOURCE_CODE_DEFAULTS
    from list_sid_target_model import TARGET_SID_DEFAULTS
    root.mkdir(parents=True, exist_ok=False)
    cfg = read(BASE / "runs/small_task_comparison_8650/protocol.json")
    for path, expected in cfg["code_hashes"].items():
        assert digest(Path(path)) == expected, path
    files = ["list_sid_story.py", "list_sid_source_model.py", "list_sid_target_model.py",
             "selftest_list_sid_source.py", "slurm_list_sid_story.sh"]
    cfg.update(TARGET_SID_DEFAULTS)
    cfg.update(tasks=TASKS, task=None, variants=VARIANTS, seeds=SEEDS, betas=BETAS,
        data_parent=str(DATA_PARENT), tokenizer_seed=TOKENIZER_SEED,
        source_code=SOURCE_CODE_DEFAULTS,
        study="two independent fixed changes on LIST v1; no parameter sweep",
        training="original2048 interaction batches INCLUDING target-only users; seed+100 dropout and seed+200 shuffle",
        source_access="only identical frozen Top100 opaque IDs/ranks; original unlabeled union vocabulary",
        target_tokenizer="per task/beta/phase ONLY all current-phase target training ratings; fixed across model seeds",
        target_tokenizer_extra_supervision="target-only rating structure contributes to collaborative codes; baseLIST uses those rows for means but zero source correction",
        target_auxiliary="mean4token CE on rows with source lists; sum divided by FULL minibatch size; eta0.1; no auxiliary inference",
        source_codes="hard3x64 compositional codes; learned cosine selectors with straight-through soft gradients; NOT cluster pooling or RQ-VAE; no compression claim",
        evaluation="all54 models and predictions frozen before test-label reads; historical splits exploratory",
        tuning="one fixed config each; only unchanged epoch0..10/global-lambda validation selection",
        slurm_job=os.environ.get("SLURM_JOB_ID"))
    cfg["code_hashes"].update({str(BASE / "scripts" / p): digest(BASE / "scripts" / p) for p in files})
    save(root / "protocol.json", cfg)
    queries, references = {}, {}
    for task, overlap in TASKS.items():
        sub = root / task
        sub.mkdir()
        save(sub / "protocol.json", dict(cfg, task=task))
        for seed in SEEDS:
            query = DATA_PARENT / task / f"seed_{seed}/queries/DNN"
            record = read(query / "complete.json")
            assert record["status"] == "PASS" and record["k"] == 100 and record["users"] == overlap
            assert digest(query / "topk_ids.npz") == record["item_ids_sha256"]
            api = np.load(query / "topk_ids.npz")
            np.testing.assert_array_equal(api["user_ids"], np.arange(overlap))
            assert api["item_ids"].shape == (overlap, 100)
            queries[f"{task}/{seed}"] = dict(path=str(query / "topk_ids.npz"), sha256=record["item_ids_sha256"])
        for beta in BETAS:
            old = DATA_PARENT / task / f"beta_{beta}"
            cell = sub / f"beta_{beta}"
            cell.mkdir()
            manifest = read(old / "manifest.json")
            assert manifest["cross_domain_raw_item_overlap"] == 0 and manifest["overlap_users"] == overlap
            for path, expected in manifest["hashes"].items():
                if path != "data/test.csv":
                    assert digest(old / path) == expected, path
            for name in ["data", "baseline", "inner_baseline"]:
                (cell / name).symlink_to((old / name).resolve(), target_is_directory=True)
            save(cell / "manifest.json", manifest)
            for seed in SEEDS:
                ref = old_baseline(task, beta, seed)
                complete = read(ref / "complete.json")
                assert digest(ref / "test.npy") == complete["test_sha256"]
                references[f"{task}/{beta}/{seed}"] = dict(path=str(ref), test_sha256=complete["test_sha256"])
                for variant in VARIANTS:
                    output(root, task, variant, beta, seed).mkdir(parents=True, exist_ok=True)
    jobs = [dict(task=t, variant=v, beta=b, seed=s)
            for s in SEEDS for t in TASKS for b in BETAS for v in VARIANTS]
    save(root / "tasks.json", jobs)
    save(root / "source_records.json", queries)
    save(root / "baseline_references.json", references)
    save(root / "data_verification.json", dict(status="PASS", model_count=len(jobs),
        no_test_labels_opened=True, identical_source_lists=True, identical_splits=True, catalogs_disjoint=True))
    print("SID_PREPARED", root, "models", len(jobs), flush=True)


def target_codes(root, task, beta, phase, data, items, cfg):
    """Atomic training-fold-only artifact cache, shared across model seeds."""
    from list_sid_target_model import fit_target_codes, target_sid_config
    out = root / "target_codes" / task / f"beta_{beta}" / phase
    out.parent.mkdir(parents=True, exist_ok=True)
    source = root / task / f"beta_{beta}/data" / ("inner_train.csv" if phase == "select" else "train.csv")
    input_hash = digest(source)
    settings = target_sid_config(cfg)
    try:
        out.mkdir()
        owner = True
    except FileExistsError:
        owner = False
    if owner:
        try:
            start = time.monotonic()
            fit = fit_target_codes(data, items, cfg, TOKENIZER_SEED, phase=phase)
            arrays = {k: v for k, v in fit.items() if isinstance(v, np.ndarray)}
            meta = fit["metadata"]
            assert arrays["codes"].shape == (items, 3)
            np.savez_compressed(out / "artifact.npz", **arrays)
            save(out / "complete.tmp.json", dict(status="PASS", phase=phase, seed=TOKENIZER_SEED,
                input_path=str(source), input_sha256=input_hash, input_rows=len(data), settings=settings,
                artifact_sha256=digest(out / "artifact.npz"), seconds=time.monotonic()-start,
                test_labels_opened=False, metadata=meta))
            (out / "complete.tmp.json").replace(out / "complete.json")
            print("TARGET_CODES_FIT", task, beta, phase, time.monotonic()-start, flush=True)
        except BaseException as exc:
            save(out / "failure.json", dict(error=repr(exc)))
            raise
    else:
        start = time.monotonic()
        while not (out / "complete.json").exists():
            if (out / "failure.json").exists():
                raise RuntimeError(read(out / "failure.json"))
            if time.monotonic() - start > 3600:
                raise TimeoutError(f"Waiting for tokenizer {out}")
            time.sleep(1)
    done = read(out / "complete.json")
    assert done["input_sha256"] == input_hash and done["seed"] == TOKENIZER_SEED
    assert done["phase"] == phase and done["settings"] == settings
    assert digest(out / "artifact.npz") == done["artifact_sha256"]
    return np.load(out / "artifact.npz")["codes"], str(out)


def make_model(variant, lists, vocab, items, base, cfg, codes=None):
    if variant == "source_sid":
        from list_sid_source_model import SourceCodeResidual
        return SourceCodeResidual(lists, vocab, items, base, cfg).to(lists.device)
    if variant == "target_sid_aux":
        from list_sid_target_model import TargetCodeResidual
        return TargetCodeResidual(lists, vocab, items, base, cfg,
                                  torch.as_tensor(codes, dtype=torch.long, device=lists.device)).to(lists.device)
    return ResidualList(lists, vocab, items, base, cfg).to(lists.device)


def train(root, task, variant, beta, seed):
    check_code(root)
    assert os.environ.get("SLURM_JOB_ID"), "GPU training requires Slurm"
    if variant == "LIST_v1":
        archived.PRIOR, archived.TASK = DATA_PARENT, task
        archived.list_run(root / task, seed, beta, "real")
        return
    student_guard(root)
    cfg = read(root / "protocol.json")
    cell = root / task / f"beta_{beta}"
    manifest = read(cell / "manifest.json")
    out = output(root, task, variant, beta, seed)
    with (out / "started.json").open("x") as handle:
        json.dump(dict(job=os.environ["SLURM_JOB_ID"], gpu=os.environ["CUDA_VISIBLE_DEVICES"]), handle)
    query_record = read(root / "source_records.json")[f"{task}/{seed}"]
    assert digest(Path(query_record["path"])) == query_record["sha256"]
    ids = np.load(query_record["path"])["item_ids"]
    vocabulary = np.unique(ids)
    compact = np.searchsorted(vocabulary, ids) + 1
    lists = torch.zeros((manifest["target_users"], cfg["k"]), device="cuda", dtype=torch.long)
    lists[:len(ids)] = torch.as_tensor(compact, device="cuda")
    validation = frame(cell / "data/validation.csv")
    vx = torch.as_tensor(validation[:, :2], device="cuda", dtype=torch.long)
    tx = torch.as_tensor(np.load(cell / "data/test_features.npy"), device="cuda", dtype=torch.long)
    selection, initial_hashes, code_records, phase_records = None, [], {}, {}
    for phase in ["select", "refit"]:
        data_path = cell / ("data/inner_train.csv" if phase == "select" else "data/train.csv")
        assert digest(data_path) == manifest["hashes"][str(data_path.relative_to(cell))]
        data = frame(data_path)
        assert not np.intersect1d(np.unique(data[:, 0]), np.load(cell / "data/test_users.npy")).size
        if phase == "select":
            assert not np.intersect1d(np.unique(data[:, 0]), np.load(cell / "data/validation_users.npy")).size
        baseline = cell / ("inner_baseline" if phase == "select" else "baseline")
        base64 = predict(dict(np.load(baseline / "statistics.npz")), np.arange(manifest["target_items"]), "item_mean")
        base = torch.as_tensor(base64.astype(np.float32), device="cuda")
        seen = torch.as_tensor(np.load(baseline / "seen.npy"), device="cuda")
        codes = None
        if variant == "target_sid_aux":
            codes, code_records[phase] = target_codes(root, task, beta, phase, data, manifest["target_items"], cfg)
        seed_all(seed)
        reference = ResidualList(lists, len(vocabulary), manifest["target_items"], base, cfg).cuda()
        common = {name: p.detach().clone() for name, p in reference.named_parameters()}
        reference_count = sum(p.numel() for p in reference.parameters())
        del reference
        seed_all(seed)
        model = make_model(variant, lists, len(vocabulary), manifest["target_items"], base, cfg, codes)
        named = dict(model.named_parameters())
        for name, value in common.items():
            actual = model.source_items.weight if name == "source_items.weight" else named[name]
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        del common, named
        initial_hashes.append(parameter_hash(model))
        if len(initial_hashes) == 2:
            assert initial_hashes[0] == initial_hashes[1], "Refit initialization differs"
        phase_records[phase] = dict(rows=len(data), source_list_rows=int((data[:, 0] < len(ids)).sum()),
            parameters=sum(p.numel() for p in model.parameters()), base_parameters=reference_count,
            common_initialization_identical=True)
        if hasattr(model, "parameter_report"):
            phase_records[phase]["parameter_report"] = model.parameter_report()
        seed_all(seed + 100)
        x = torch.as_tensor(data[:, :2], device="cuda", dtype=torch.long)
        y = torch.as_tensor(data[:, 2], device="cuda", dtype=torch.float32)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                    lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        generator = torch.Generator(device="cuda").manual_seed(seed + 200)
        if phase == "select":
            np.testing.assert_array_equal(corrections(model, vx, seen), np.zeros(len(vx), dtype=np.float32))
            baseval = base64[validation[:, 1].astype(int)]
            selection = dict(epoch=0, lambda_value=0.,
                validation_rmse=float(np.sqrt(np.square(baseval-validation[:, 2]).mean())))
            torch.save(model.state_dict(), out / "selected.pt")
            np.save(out / "validation.npy", baseval)
        epochs = cfg["epochs"] if phase == "select" else selection["epoch"]
        history = []
        for epoch in range(1, epochs + 1):
            tick = time.monotonic()
            model.train()
            totals = {}
            for ix in torch.randperm(len(x), device="cuda", generator=generator).split(cfg["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss, parts = model.training_loss(x[ix, 0], x[ix, 1], y[ix])
                loss.backward()
                optimizer.step()
                for key, value in parts.items():
                    if key != "aux_rows" and isinstance(value, torch.Tensor) and value.numel() == 1:
                        totals[key] = totals.get(key, torch.zeros((), device="cuda")) + value.detach()*len(ix)
            row = dict(epoch=epoch, seconds=time.monotonic()-tick,
                       **{key: float(value.item()/len(x)) for key, value in totals.items()})
            assert all(np.isfinite(v) for v in row.values()), row
            if phase == "select":
                correction = corrections(model, vx, seen)
                row["validation"] = {}
                for lam in cfg["lambdas"]:
                    prediction = baseval + lam*correction
                    rmse = float(np.sqrt(np.square(prediction-validation[:, 2]).mean()))
                    row["validation"][str(lam)] = rmse
                    if rmse < selection["validation_rmse"]:
                        selection = dict(epoch=epoch, lambda_value=lam, validation_rmse=rmse)
                        torch.save(model.state_dict(), out / "selected.pt")
                        np.save(out / "validation.npy", prediction)
            history.append(row)
            save(out / f"{phase}_history.json", history)
            print("SID_EPOCH", task, beta, seed, variant, phase, json.dumps(row), flush=True)
        if phase == "select":
            model.load_state_dict(torch.load(out / "selected.pt", map_location="cuda", weights_only=True))
            np.testing.assert_array_equal(baseval + selection["lambda_value"]*corrections(model, vx, seen),
                                          np.load(out / "validation.npy"))
            save(out / "selection.json", selection)
        else:
            save(out / "refit_history.json", history)
            torch.save(model.state_dict(), out / "final.pt")
            prediction = base64[tx[:, 1].cpu().numpy()] + selection["lambda_value"]*corrections(model, tx, seen)
            assert np.isfinite(prediction).all()
            np.save(out / "test.npy", prediction)
            restored = make_model(variant, lists, len(vocabulary), manifest["target_items"], base, cfg, codes)
            restored.load_state_dict(torch.load(out / "final.pt", map_location="cuda", weights_only=True))
            np.testing.assert_array_equal(base64[tx[:, 1].cpu().numpy()] +
                selection["lambda_value"]*corrections(restored, tx, seen), prediction)
            if variant == "source_sid":
                np.savez_compressed(out / "source_codes.npz", raw_ids=vocabulary,
                                    codes=model.export_codes().numpy()[1:])
            extra = {"source_codes_sha256": digest(out / "source_codes.npz")} if variant == "source_sid" else {}
            save(out / "complete.json", dict(status="PASS", task=task, variant=variant, beta=beta, seed=seed,
                selection_sha256=digest(out / "selection.json"), final_sha256=digest(out / "final.pt"),
                test_sha256=digest(out / "test.npy"), initial_parameters_sha256=initial_hashes[0],
                query_sha256=query_record["sha256"], phase_records=phase_records, target_code_records=code_records,
                full_outer_train_rows=len(data), refit_epochs=epochs, refit_reinitialized=True,
                restored_predictions=True, test_labels_opened=False, **extra))
            del restored
        del model, optimizer, x, y, data
    print("SID_FROZEN", task, beta, seed, variant, flush=True)


def worker(root, index):
    check_code(root)
    for job in read(root / "tasks.json"):
        out = output(root, **job)
        try:
            with (out / "claim.json").open("x") as handle:
                json.dump(dict(worker=index, gpu=os.environ["CUDA_VISIBLE_DEVICES"]), handle)
        except FileExistsError:
            continue
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "train", "--root", str(root)]
        for key, value in job.items():
            command += ["--" + key, str(value)]
        print("SID_TASK", index, job, flush=True)
        with (out / "train.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            save(out / "failure.json", dict(returncode=result.returncode, task=job))
            raise RuntimeError(f"Training failed: {out}/train.log")


def metrics(pred, y):
    error = pred-y
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(root):
    check_code(root)
    assert read(root / "selftest.json")["status"] == "PASS"
    frozen = {}
    for job in read(root / "tasks.json"):
        out = output(root, **job)
        done = read(out / "complete.json")
        assert done["status"] == "PASS" and done["test_labels_opened"] is False
        for name, key in [("final.pt", "final_sha256"), ("test.npy", "test_sha256"), ("selection.json", "selection_sha256")]:
            assert digest(out / name) == done[key]
        if "source_codes_sha256" in done:
            assert digest(out / "source_codes.npz") == done["source_codes_sha256"]
        frozen[str(out.relative_to(root))] = digest(out / "complete.json")
    assert len(frozen) == 54
    tokenizer_records = list((root / "target_codes").glob("*/*/*/complete.json"))
    assert len(tokenizer_records) == 12
    for record in tokenizer_records:
        token = read(record)
        assert digest(record.parent / "artifact.npz") == token["artifact_sha256"]
        assert digest(Path(token["input_path"])) == token["input_sha256"]
    with (root / "evaluation_started.json").open("x") as handle:
        json.dump(frozen, handle, indent=2)
    rows, aggregates, replay = [], [], []
    for task in TASKS:
        for beta in BETAS:
            cell = root / task / f"beta_{beta}"
            manifest = read(cell / "manifest.json")
            assert digest(cell / "data/test.csv") == manifest["hashes"]["data/test.csv"]
            data = frame(cell / "data/test.csv")
            np.testing.assert_array_equal(data[:, :2], np.load(cell / "data/test_features.npy"))
            y = data[:, 2]
            mean = predict(dict(np.load(cell / "baseline/statistics.npz")), data[:, 1].astype(int), "item_mean")
            bases = {seed: metrics(np.load(output(root, task, "LIST_v1", beta, seed) / "test.npy"), y) for seed in SEEDS}
            for seed in SEEDS:
                old = old_baseline(task, beta, seed)
                current = output(root, task, "LIST_v1", beta, seed)
                replay.append(dict(task=task, beta=beta, seed=seed,
                    max_prediction_difference=float(np.max(np.abs(np.load(old / "test.npy")-np.load(current / "test.npy")))),
                    same_selection=read(old / "selection.json") == read(current / "selection.json")))
            for variant in ["item_mean", *VARIANTS]:
                group = []
                for seed in SEEDS:
                    if variant == "item_mean":
                        values, choice = metrics(mean, y), dict(epoch=0, lambda_value=0.)
                    else:
                        out = output(root, task, variant, beta, seed)
                        values, choice = metrics(np.load(out / "test.npy"), y), read(out / "selection.json")
                    row = dict(task=task, beta=beta, variant=variant, seed=seed,
                               epoch=choice["epoch"], lambda_value=choice["lambda_value"], **values)
                    for key in ["mae", "rmse"]:
                        row["delta_"+key] = values[key]-bases[seed][key]
                    rows.append(row)
                    group.append(row)
                aggregate = dict(task=task, beta=beta, variant=variant)
                for key in ["mae", "rmse", "delta_mae", "delta_rmse"]:
                    aggregate[key] = float(np.mean([r[key] for r in group]))
                    aggregate[key+"_std"] = float(np.std([r[key] for r in group], ddof=1))
                aggregate["rmse_improved_seeds"] = sum(r["delta_rmse"] < 0 for r in group)
                aggregate["mean_fallback_seeds"] = sum(r["lambda_value"] == 0 for r in group)
                aggregates.append(aggregate)
    write_csv(root / "per_seed.csv", rows)
    write_csv(root / "comparison.csv", aggregates)
    save(root / "baseline_replay.json", replay)
    save(root / "summary.json", dict(status="PASS", comparison=aggregates, per_seed=rows, baseline_replay=replay))
    save(root / "verification.json", dict(status="PASS", completed_models=54,
        all_frozen_before_test_evaluation=True, checkpoints_predictions_and_selection_verified=True,
        baseline_replay_max_difference=max(r["max_prediction_difference"] for r in replay),
        baseline_replay_all_same_selection=all(r["same_selection"] for r in replay),
        catalogs_disjoint=True, code_hashes_verified=True))
    publish_report(root, aggregates, rows, replay)
    print("SID_RESULTS", json.dumps(aggregates), flush=True)


def publish_report(root, aggregates, rows, replay):
    names = {"item_mean": "普通均值", "LIST_v1": "原版LIST v1（本轮重跑）",
             "source_sid": "A：源物品组合编码", "target_sid_aux": "B：目标编码生成监督"}
    tasks = {"music_movie_clean": "Music→Movie去重", "book_movie": "Book→Movie", "book_music": "Book→Music"}
    lines = ["# LIST两条SID路线：固定配置三数据集实验", "",
        f"Slurm作业：{read(root / 'protocol.json')['slurm_job']}。结果目录：{root.relative_to(BASE)}。没有新参数搜索。",
        "", "三个种子2020/2021/2022，β20/80；每个单模型先计算指标，再取均值与样本标准差。负差值表示比同批原版LIST更好。",
        "", "| 任务 | β | 方法 | MAE ± SD | RMSE ± SD | ΔMAE | ΔRMSE | RMSE改善种子数 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in aggregates:
        lines.append(f"| {tasks[r['task']]} | {r['beta']}% | {names[r['variant']]} | {r['mae']:.6f} ± {r['mae_std']:.6f} | {r['rmse']:.6f} ± {r['rmse_std']:.6f} | {r['delta_mae']:+.6f} | {r['delta_rmse']:+.6f} | {r['rmse_improved_seeds']}/3 |")
    lines += ["", "## 本轮实际改动", "",
        "- A：源ID保留32维可学习选择向量，选择3个码本中各1个代码；每个码本64项、每项32维。前向为硬组合代码，反向通过温度0.5的softmax直通梯度学习分配；训练批次代码均衡KL权重0.01。它不是旧K-means分簇聚合，不是RQ-VAE，也不宣称训练参数压缩。",
        "- B：仅用当前训练阶段的全部目标训练评分构造残差协同矩阵，经SVD32与3层64项量化得到固定目标代码；选择与重训分别拟合，三个模型种子共享各自训练折的固定码本。小GRU64预测3个目标代码及高于/等于/低于物品均值的方向；辅助CE权重0.1。只有有源列表的训练行接受辅助监督，损失按完整批大小归一化。预测继续使用原LIST评分头。",
        "- B码本利用了目标专属用户的训练评分结构；这在目标端访问条件内，但原版主要通过这些记录计算均值，因此不是只有参数化方式发生变化。",
        "", "## 共同对照条件", "",
        "冻结源DNN Top100和去重数据划分保持原样；32→64→32编码器、内积、普通物品均值、零残差初始化、Adam0.001、weight decay1e-5、dropout0.1、2048交互批、最多10轮均沿用原版。包括没有源列表的目标专属用户训练行，未改成按用户批处理。",
        "仅沿用原版验证规则选择epoch0..10和全局λ∈{0,.25,.5,.75,1}，选后从相同初始参数重训；新增方法没有超参数搜索。54个模型及测试预测全部冻结后，本轮才读取测试标签。测试划分已用于历史探索，结果不代表独立新测试集。",
        f"原版重跑与历史预测的最大绝对差：{max(r['max_prediction_difference'] for r in replay):.9g}；所有选择相同：{all(r['same_selection'] for r in replay)}。",
        "", "## 逐种子选模", "",
        "| 任务 | β | 方法 | 种子 | epoch | λ | MAE | RMSE | ΔRMSE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r["variant"] != "item_mean":
            lines.append(f"| {tasks[r['task']]} | {r['beta']}% | {names[r['variant']]} | {r['seed']} | {r['epoch']} | {r['lambda_value']} | {r['mae']:.6f} | {r['rmse']:.6f} | {r['delta_rmse']:+.6f} |")
    report = "\n".join(lines) + "\n"
    (root / "report.md").write_text(report)
    (BASE / "docs/list_sid_story_experiment.md").write_text(report)
    write_csv(BASE / "docs/list_sid_story_results.csv", aggregates)
    write_csv(BASE / "docs/list_sid_story_per_seed_results.csv", rows)


def selftest(root):
    check_code(root)
    from selftest_list_sid_source import main as source_test
    from list_sid_target_model import selftest as target_test
    source_test()
    target = target_test()
    cfg = read(root / "protocol.json")
    lists = torch.as_tensor([list(range(1, 101)), list(range(100, 0, -1)), [0]*100], device="cuda")
    base = torch.tensor([3., 4., 2.], device="cuda")
    users = torch.arange(3, device="cuda")
    for variant in ["source_sid", "target_sid_aux"]:
        seed_all(2020)
        model = make_model(variant, lists, 100, 3, base, cfg, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        model.eval()
        torch.testing.assert_close(model(users, users), base, rtol=0, atol=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        for _ in range(2):
            model.train()
            optimizer.zero_grad()
            loss, _ = model.training_loss(users, users, base+1)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        model.eval()
        assert model(users, users)[2] == base[2]
    save(root / "selftest.json", dict(status="PASS", source_cpu="PASS", target_cpu=target, cuda="PASS"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "selftest", "train", "worker", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--beta", type=int, choices=BETAS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "train":
        train(root, args.task, args.variant, args.beta, args.seed)
    elif args.stage == "worker":
        worker(root, args.worker)
    else:
        globals()[args.stage](root)
