#!/usr/bin/env python3
"""Summarize archived LIST results on CPU; never train or modify run artifacts."""

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/list_optimization_overview.md"
SOURCES = {}
RECORDS = []
CHECKS = []
BASE = "small_task_comparison_8650"
TASK_NAMES = {
    "music_movie_clean": "Music→Movie（去重）",
    "book_movie": "Book→Movie",
    "book_music": "Book→Music",
}


def read(path):
    path = ROOT / path
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8-sig")


def rows(run, name="comparison.csv"):
    return list(csv.DictReader(read(f"runs/{run}/{name}").splitlines()))


def obj(path):
    return json.loads(read(path))


def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=0), (a, b)


def unique(items, **filters):
    selected = [r for r in items if all(str(r.get(k)) == str(v) for k, v in filters.items())]
    assert len(selected) == 1, (filters, len(selected))
    return selected[0]


def add(task, beta, method, label, seed, mae, rmse, source, variant,
        selection="fixed_variant", **extra):
    RECORDS.append(dict(task=task, beta=int(beta), method=method, label=label,
                        seed=str(seed), mae=float(mae), rmse=float(rmse),
                        source=source, original_variant=variant,
                        selection=selection, **extra))


# Every fixed variant is retained, including controls. No test-score selection.
# method, label, run, per-seed selector, aggregate selector
SPECS = [
    ("old_list", "早期LIST（非物品均值残差）", BASE, {"method": "LIST_original"}, {"method": "LIST_original"}),
    ("list_v1", "原版LIST v1", BASE, {"method": "LIST_residual_real"}, {"method": "LIST_residual_real"}),
    ("inverse_degree", "频次降权", "list_weighting_8651", {"variant": "inverse_degree", "arm": "real"}, {"method": "inverse_degree_real"}),
    ("target_attention", "目标相关加权", "list_weighting_8651", {"variant": "target_attention", "arm": "real"}, {"method": "target_attention_real"}),
    ("segments", "排名分段：3段", "list_segmented_8703", {"variant": "rank_segments", "arm": "real"}, {"method": "rank_segments_real"}),
    ("random_segments", "分段对照：随机3组", "list_segmented_8703", {"variant": "random_groups", "arm": "real"}, {"method": "random_groups_real"}),
    ("cluster3", "共现聚类：C=3", "list_cluster_8708", {"variant": "cooccurrence", "arm": "real"}, {"method": "cooccurrence_real"}),
    ("random_clusters", "聚类对照：同大小随机簇", "list_cluster_8708", {"variant": "size_matched_random", "arm": "real"}, {"method": "size_matched_random_real"}),
    ("cluster_selected", "簇数搜索：验证选C=1", "list_cluster_search_8711", {"method": "C1_cooccurrence_real"}, {"method": "C1_cooccurrence_real"}),
    ("ib_noise", "IB结构对照：η=0", "list_ib_story_8763", {"variant": "noise_only"}, {"variant": "noise_only"}),
    ("ib_small", "IB：η=10⁻⁴", "list_ib_story_8763", {"variant": "ib_1e4"}, {"variant": "ib_1e4"}),
    ("ib_large", "IB：η=10⁻³", "list_ib_story_8763", {"variant": "ib_1e3"}, {"variant": "ib_1e3"}),
    ("memory20", "Engram启发：前20名组合记忆", "list_memory_story_8760", {"variant": "pair20"}, {"variant": "pair20"}),
    ("memory50", "Engram启发：前50名组合记忆", "list_memory_story_8760", {"variant": "pair50"}, {"variant": "pair50"}),
    ("k20", "固定列表长度：K=20", "list_acquisition_story_8763", {"variant": "k_20"}, {"variant": "k_20"}),
    ("k50", "固定列表长度：K=50", "list_acquisition_story_8763", {"variant": "k_50"}, {"variant": "k_50"}),
    ("k200", "固定列表长度：K=200", "list_acquisition_story_8763", {"variant": "k_200"}, {"variant": "k_200"}),
    ("point_matched", "歧义建模对照：按用户批处理的点模型", "list_ambiguity_8913", {"variant": "point_matched"}, {"variant": "point_matched"}),
    ("gaussian_fixed", "条件高斯：固定std=0.3", "list_ambiguity_8913", {"variant": "gaussian_fixed"}, {"variant": "gaussian_fixed"}),
    ("gaussian_default", "条件高斯：学习方差（默认）", "list_ambiguity_8913", {"variant": "gaussian_learned"}, {"variant": "gaussian_learned"}),
    ("energy20", "条件能量：训练/预测各20步", "list_ambiguity_8913", {"variant": "energy_20"}, {"variant": "energy_20"}),
    ("energy60", "条件能量：训练/预测各60步", "list_ambiguity_8913", {"variant": "energy_60"}, {"variant": "energy_60"}),
]


def load_common():
    for beta in [20, 80]:
        r = unique(rows(BASE), beta=beta, method="item_mean")
        for seed in [2020, 2021, 2022]:
            add("music_movie_clean", beta, "mean", "普通物品均值", seed, r["mae"], r["rmse"],
                f"runs/{BASE}/comparison.csv", "item_mean", "deterministic_statistic")
    for method, label, run, selector, aggregate_selector in SPECS:
        data, aggregates = rows(run, "per_seed.csv"), rows(run)
        for beta in [20, 80]:
            part = [unique(data, beta=beta, seed=s, **selector) for s in [2020, 2021, 2022]]
            expected = unique(aggregates, beta=beta, **aggregate_selector)
            for metric in ["mae", "rmse"]:
                close(statistics.mean(float(r[metric]) for r in part), expected[metric])
                close(statistics.stdev(float(r[metric]) for r in part), expected[metric + "_std"])
            CHECKS.append(f"aggregate_matches_three_seeds:{method}:beta{beta}")
            for r in part:
                add("music_movie_clean", beta, method, label, r["seed"], r["mae"], r["rmse"],
                    f"runs/{run}/per_seed.csv", r.get("variant", r.get("method")),
                    "historical_validation_cluster_choice" if method == "cluster_selected" else "fixed_variant",
                    epoch=r.get("epoch", ""), lambda_value=r.get("lambda_value", ""))


def load_transfer():
    run = "list_ambiguity_transfer_8923"
    mapping = {"LIST_v1": ("list_v1", "原版LIST v1"), "item_mean": ("mean", "普通物品均值"),
               "point_matched": ("point_matched", "按用户批处理的点模型"),
               "gaussian_learned": ("gaussian_default", "条件高斯：学习方差（默认）"),
               "energy_20": ("energy20", "条件能量：训练/预测各20步")}
    data = rows(run, "per_seed.csv")
    for r in data:
        method, label = mapping[r["variant"]]
        add(r["task"], r["beta"], method, label, r["seed"], r["mae"], r["rmse"],
            f"runs/{run}/per_seed.csv", r["variant"], epoch=r["epoch"], lambda_value=r["lambda_value"])
    for a in rows(run):
        part = [r for r in data if all(r[k] == a[k] for k in ["task", "beta", "variant"])]
        assert len(part) == 3
        for metric in ["mae", "rmse"]:
            close(statistics.mean(float(r[metric]) for r in part), a[metric])
            close(statistics.stdev(float(r[metric]) for r in part), a[metric + "_std"])
    CHECKS.append("transfer_aggregates_match_three_seeds:20_groups")


def load_tuned():
    run = "list_gaussian_noise_8952"
    for r in rows(run):
        if r["method"] == "gaussian_noise" and r["validation_selected"].lower() == "true":
            add(r["task"], r["beta"], "gaussian_noise_tuned", "高斯：仅搜索评分噪声方差", r["seed"],
                r["mae"], r["rmse"], f"runs/{run}/comparison.csv", "gaussian_noise",
                "historical_validation_noise_choice", parameters=f"observation_variance={r['observation_variance']}",
                epoch=r["epoch"], lambda_value=r["lambda_value"], validation_rmse=r["validation_rmse"])
    run = "list_gaussian_sequential_8963"
    for r in rows(run):
        if r["validation_selected"].lower() == "true" and r["round"] in ["1", "3"]:
            method = "gaussian_init_tuned" if r["round"] == "1" else "gaussian_tuned"
            label = "高斯：仅搜索初始std" if r["round"] == "1" else "高斯：顺序搜索最终配置"
            add(r["task"], r["beta"], method, label, r["seed"], r["mae"], r["rmse"],
                f"runs/{run}/comparison.csv", f"round_{r['round']}", "historical_validation_sequential_choice",
                parameters=f"std_init={r['latent_std']}; lower={r['std_lower']}; upper={r['std_upper']}; observation_variance=1",
                epoch=r["epoch"], lambda_value=r["lambda_value"], validation_rmse=r["validation_rmse"])
    # Verify that all later tuning baselines are the same seed-2020 models.
    for r in rows(run, "baselines.csv"):
        m = {"LIST_v1": "list_v1", "gaussian_previous": "gaussian_default", "item_mean": "mean"}[r["method"]]
        original = unique(RECORDS, task=r["task"], beta=r["beta"], method=m, seed=2020)
        close(original["mae"], r["mae"])
        close(original["rmse"], r["rmse"])
    run = "list_langevin_sensitivity_8968"
    data = rows(run, "results.csv")
    selected = obj(f"runs/{run}/review_best_single_changes.json")
    for choice in selected:
        candidates = [r for r in data if r["task"] == choice["task"] and int(r["beta"]) == choice["beta"]]
        # Each default is repeated once per axis. De-duplicate, then use historical tie order.
        by_case = {int(r["case_id"]): r for r in candidates}
        assert len(by_case) == 15
        winner = min(by_case.values(), key=lambda r: (float(r["validation_rmse"]), int(r["case_id"])))
        assert int(winner["case_id"]) == choice["case_id"]
        r = choice["result"]
        for metric in ["mae", "rmse", "validation_rmse"]:
            close(r[metric], winner[metric])
        add(r["task"], r["beta"], "energy_tuned", "朗之万：验证选中的单项改动", r["seed"],
            r["mae"], r["rmse"], f"runs/{run}/review_best_single_changes.json", f"case_{r['case_id']}",
            "review_min_validation_among_existing_15_one_factor_configs", parameters=choice["change"],
            epoch=r["epoch"], lambda_value=r["lambda_value"], validation_rmse=r["validation_rmse"])
        for method, prefix in [("list_v1", "list"), ("energy20", "energy"), ("mean", "mean")]:
            original = unique(RECORDS, task=r["task"], beta=r["beta"], method=method, seed=2020)
            for metric in ["mae", "rmse"]:
                close(original[metric], choice["baseline"][prefix + "_" + metric])
    CHECKS.append("tuning_baselines_match_seed2020:all_tasks")
    CHECKS.append("langevin_choices_use_validation_only:6_tasks")


def verify_provenance():
    runs = sorted({r["source"].split("/")[1] for r in RECORDS})
    base_hash = hashlib.sha256((ROOT / f"runs/{BASE}/protocol.json").read_bytes()).hexdigest()
    for run in runs:
        v = obj(f"runs/{run}/verification.json")
        assert v["status"] == "PASS", run
        protocol = obj(f"runs/{run}/protocol.json")
        if run in {s[2] for s in SPECS[2:17]}:
            # Historical protocols retain server paths; the archive may be cloned elsewhere.
            assert Path(protocol["parent"]).parts[-2:] == ("runs", BASE)
            assert protocol["parent_protocol_sha256"] == base_hash
        CHECKS.append(f"archived_verification_PASS:{run}")
    # Distribution models copy manifests; other architecture trials reuse the parent directly.
    for beta in [20, 80]:
        base = obj(f"runs/{BASE}/beta_{beta}/manifest.json")
        assert base["overlap_users"] == 17904 and base["cross_domain_raw_item_overlap"] == 0
        for run in ["list_ambiguity_8913", "list_gaussian_noise_8952", "list_gaussian_sequential_8963", "list_langevin_sensitivity_8968"]:
            prefix = "" if run == "list_ambiguity_8913" else "music_movie_clean/"
            manifest = obj(f"runs/{run}/{prefix}beta_{beta}/manifest.json")
            assert manifest["hashes"] == base["hashes"], (run, beta)
        CHECKS.append(f"identical_manifest_data_and_mean_hashes:music_movie_clean:beta{beta}")


def table(header, data):
    align = ["---:" if "MAE" in h or "RMSE" in h else "---" for h in header]
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in data])


def aggregate(task, beta, method):
    part = [r for r in RECORDS if r["task"] == task and r["beta"] == beta and r["method"] == method]
    assert {r["seed"] for r in part} == {"2020", "2021", "2022"}
    return {m: statistics.mean(r[m] for r in part) for m in ["mae", "rmse"]}


def pair(r):
    return f"{r['mae']:.6f} / {r['rmse']:.6f}"


def common_table(seed=None):
    methods = [("mean", "普通物品均值")] + [(s[0], s[1]) for s in SPECS]
    if seed is not None:
        methods = [(m, l) for m, l in methods if m not in {"old_list", "random_segments", "random_clusters"}]
        methods += [("gaussian_noise_tuned", "高斯：仅搜索评分噪声方差"),
                    ("gaussian_init_tuned", "高斯：仅搜索初始std"),
                    ("gaussian_tuned", "高斯：顺序搜索最终配置"),
                    ("energy_tuned", "朗之万：验证选中的单项改动")]
    output = []
    for method, label in methods:
        cells = ["**" + label + "**" if method == "list_v1" else label]
        for beta in [20, 80]:
            get = (lambda m: aggregate("music_movie_clean", beta, m)) if seed is None else (
                lambda m: unique(RECORDS, task="music_movie_clean", beta=beta, method=m, seed=seed))
            r, b = get(method), get("list_v1")
            cells += [pair(r), f"{r['rmse'] - b['rmse']:+.6f}"]
        output.append(cells)
    return table(["版本", "β20 MAE / RMSE", "ΔRMSE 对v1", "β80 MAE / RMSE", "ΔRMSE 对v1"], output)


def multi_task_table(tuned=False):
    output = []
    methods = ["mean", "list_v1", "gaussian_default", "energy20"]
    labels = ["普通均值", "原版LIST v1", "默认学习方差高斯", "默认20步朗之万"]
    if tuned:
        methods = ["list_v1", "gaussian_tuned", "energy_tuned"]
        labels = ["原版LIST v1", "高斯顺序搜索最终配置", "朗之万验证选单项改动"]
    for task in TASK_NAMES:
        for beta in [20, 80]:
            get = (lambda m: unique(RECORDS, task=task, beta=beta, method=m, seed=2020)) if tuned else (
                lambda m: aggregate(task, beta, m))
            cells = [TASK_NAMES[task], f"{beta}%"] + [pair(get(m)) for m in methods]
            if tuned:
                cells += [f"{get(m)['rmse'] - get('list_v1')['rmse']:+.6f}" for m in ["gaussian_tuned", "energy_tuned"]]
            output.append(cells)
    return table(["任务", "β"] + [s + " MAE / RMSE" for s in labels]
                 + (["高斯ΔRMSE", "朗之万ΔRMSE"] if tuned else []), output)


def parameter_table():
    output = []
    energy_labels = {"default": "默认；最终退回均值", "train_mala_step=0.1": "训练步长0.1",
                     "energy_bound=10.0": "能量上界10", "train_chains=64": "训练链数64",
                     "eval_steps=60": "预测采样步数60"}
    for task in TASK_NAMES:
        for beta in [20, 80]:
            g = unique(RECORDS, task=task, beta=beta, method="gaussian_tuned", seed=2020)
            e = unique(RECORDS, task=task, beta=beta, method="energy_tuned", seed=2020)
            params = dict(x.split("=") for x in g["parameters"].split("; "))
            output.append([TASK_NAMES[task], f"{beta}%", " / ".join(params[k] for k in ["std_init", "lower", "upper"]),
                           energy_labels[e["parameters"]]])
    return table(["任务", "β", "高斯：初始std / 下限 / 上限", "朗之万历史验证选择"], output)


def write_csv(name, data):
    fields = list(dict.fromkeys(k for r in data for k in r))
    with (ROOT / "docs" / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(data)


def main():
    load_common()
    load_transfer()
    load_tuned()
    verify_provenance()
    keys = [(r["task"], r["beta"], r["method"], r["seed"]) for r in RECORDS]
    assert len(keys) == len(set(keys))
    for r in RECORDS:
        b = unique(RECORDS, task=r["task"], beta=r["beta"], method="list_v1", seed=r["seed"])
        r["delta_mae_vs_list_v1"] = r["mae"] - b["mae"]
        r["delta_rmse_vs_list_v1"] = r["rmse"] - b["rmse"]
    write_csv("list_optimization_overview_results.csv", RECORDS)
    aggregates = []
    for task, beta, method in sorted({(r["task"], r["beta"], r["method"]) for r in RECORDS}):
        part = [r for r in RECORDS if (r["task"], r["beta"], r["method"]) == (task, beta, method)]
        if len(part) != 3:
            continue
        r = {"task": task, "beta": beta, "method": method, "label": part[0]["label"], "seeds": "2020,2021,2022"}
        for metric in ["mae", "rmse"]:
            r[metric] = statistics.mean(x[metric] for x in part)
            r[metric + "_std"] = statistics.stdev(x[metric] for x in part)
            r["delta_" + metric + "_vs_list_v1"] = statistics.mean(x["delta_" + metric + "_vs_list_v1"] for x in part)
        r["rmse_seed_wins_vs_list_v1"] = sum(x["delta_rmse_vs_list_v1"] < 0 for x in part)
        aggregates.append(r)
    write_csv("list_optimization_overview_3seed.csv", aggregates)
    blocks = {"COMMON_3SEED": common_table(), "COMMON_SEED2020": common_table(2020),
              "TRANSFER_3SEED": multi_task_table(), "TRANSFER_TUNED": multi_task_table(True),
              "TUNED_PARAMETERS": parameter_table()}
    text = DOC.read_text()
    for name, content in blocks.items():
        pattern = rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)"
        text, n = re.subn(pattern, lambda m: m[1] + content + "\n" + m[2], text, flags=re.S)
        assert n == 1, (name, n)
    DOC.write_text(text)
    audit = dict(status="PASS", scope="CPU summary only; archived verification reports read, no new prediction audit or training",
                 record_count=len(RECORDS), three_seed_groups=len(aggregates), checks=CHECKS,
                 input_sha256=SOURCES, outputs=[str(DOC.relative_to(ROOT)),
                     "docs/list_optimization_overview_results.csv", "docs/list_optimization_overview_3seed.csv"])
    (ROOT / "docs/list_optimization_overview_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ["status", "record_count", "three_seed_groups"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
