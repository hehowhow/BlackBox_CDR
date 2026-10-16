# BlackBox CDR — LAPT / LIST 跨域冷用户评分预测

源推荐服务只提供 Top-K 物品 ID 和排名时，能否帮助预测另一个域的评分？
本仓库保存实验代码、固定参数、两卡 Slurm 入口、结果表与审计记录。

**`lapt` 分支：LAPT（Limited-Access Preference Transfer），截至 2026-10-16。**
当前方法接收源域 Top-100 原物品 ID 与排名，通过差异化偏好增强（DPE）提取列表表示，
再用条件能量驱动的偏好采样（CEPS）预测目标物品普通均值之上的评分修正。
这一版不使用粗码，也不按物品出现频次直接加权。

[LAPT 实现、参数与运行说明](docs/lapt.md) ·
[论文 outline](docs/lapt_paper_outline.md) ·
[最新 MF 54 组结果与分析](docs/lapt_mf_full_seed2020.md) ·
[PDF](docs/lapt_mf_full_seed2020.pdf) ·
[LaTeX](docs/lapt_mf_full_seed2020.tex) ·
[完整指标 CSV](docs/lapt_mf_full_seed2020_results.csv)。

本次 MF 实验覆盖三个任务、β20/50/80、seed2020、六种方法；此前 DNN seed2021
实验另行保留。两轮结果不能混成多种子均值；LAPT 的改善没有覆盖全部设置。
新环境可以先运行文档中的 CPU 自检；全量训练仍需重建未公开的数据与历史缓存，
并调整服务器路径。模型、训练和采样实现保留实验时版本。

## 历史 LIST 与优化归档

**前序 `optimization4metric` 分支：LIST额外优化实验归档（截至2026-09-27）。**
[全部优化的目的、结论与统一对照表](docs/list_optimization_overview.md) ·
[逐种子及调参结果CSV](docs/list_optimization_overview_results.csv) ·
[三种子均值与标准差CSV](docs/list_optimization_overview_3seed.csv)。
涵盖加权降权、分段、聚类及簇数搜索、IB、Engram启发组合记忆、固定Top-K、
条件高斯、条件能量/朗之万及其参数扫描。三种子平均与seed2020调参结果分别报告；
这些是各自独立的变体，没有合并成一个新主模型。

**归档 LIST V1：DNN 源 Top100 → 排名加权聚合 → MLP → 普通目标物品均值上的残差修正。**
DNN 沿用 PFCDR 作者命名，实际为用户仿射变换后与物品向量内积。
学生不读取源评分、教师参数或返回分数；需要已知的跨域用户对应与重叠训练用户。

[当前版本与参数](docs/current_list_version.md) ·
[主版本论文outline（与主图一致）](docs/list_residual_v1_paper_outline.md) ·
[参数快照 JSON](configs/list_residual_v1.json) ·
[完整结果与分析](runs/list_weighting_8651/report.md) ·
[对比表 CSV](runs/list_weighting_8651/main_comparison.csv) ·
[剩余优化故事线与当前优先级](docs/list_remaining_optimization_tracks.md) ·
[IB/组合记忆原型与固定K对照结果](docs/list_three_story_experiments.md) ·
[原方案实施核对与更正](docs/list_implementation_alignment_audit_20260925.md)

## 当前结果

Music→Movie 去重版，17,904 名重叠用户，三个模型种子 2020/2021/2022。
β 为重叠用户中目标冷启动测试比例；下表为各模型指标均值，越低越好，不是预测集成。
所有方法使用相同目标训练/测试划分；普通物品均值不平滑。

| 方法 | β20% MAE | β20% RMSE | β80% MAE | β80% RMSE |
| --- | ---: | ---: | ---: | ---: |
| 普通物品均值 | 0.828612 | 1.074618 | 0.840744 | 1.097641 |
| EMCDR–MF | 1.315861 | 1.719547 | 1.352129 | 1.779450 |
| EMCDR–DNN | 0.819904 | 1.069249 | 0.826689 | 1.097017 |
| PFCDR–MF | 0.992507 | 1.262861 | 1.048391 | 1.391033 |
| PFCDR–DNN | 0.817090 | 1.073654 | 0.849226 | 1.140535 |
| 原LIST（DNN源列表） | 0.822822 | 1.067575 | 0.870605 | 1.124753 |
| 均值＋LIST修正（第一项） | 0.808497 | 1.056126 | 0.833717 | 1.089396 |
| 修正＋频次降权（第三项A） | 0.808067 | 1.058466 | 0.833007 | 1.089006 |
| 修正＋目标相关加权（第三项B） | 0.814642 | 1.056793 | 0.833733 | 1.090879 |

当前主版本在两档的 MAE/RMSE 均超过普通均值、原 LIST 和完整重训的错配列表对照。
两种加权没有一致改善主版本，所以保留为消融。β80% 的最低 MAE 属于 EMCDR–DNN，
不能声称 LIST 在全部指标上最优。置信区间、标准差和逐种子结果见完整报告。

上表对应去重Music→Movie两档。后续已经在Book→Movie、Book→Music的β20/80重跑v1，
并与高斯及条件能量比较，见[三任务扩展](docs/list_ambiguity_transfer_experiment.md)。
早期其他方向的旧LIST结果不等于v1效果。历史测试已用于探索，不是全新盲测。PFCDR 没有做 K/LR 搜索，
这些结果不是作者十次运行的逐参数复现，也不是各方法充分调参后的性能上限。

## 代码与结果入口

| 内容 | 实现/协议 | 结果 |
| --- | --- | --- |
| 三个论文方向、β20/50/80、DNN 原 LIST | [paper_list_dnn.py](scripts/paper_list_dnn.py)、[协议](docs/paper_list_dnn.md) | [Slurm 8638](runs/paper_list_dnn_8638/report.md) |
| 当前均值残差 LIST、错配对照及 EMCDR/PFCDR | [small_task_comparison.py](scripts/small_task_comparison.py)、[协议](docs/small_task_comparison.md) | [Slurm 8650](runs/small_task_comparison_8650/report.md) |
| 频次降权、目标相关加权 | [list_weighting.py](scripts/list_weighting.py)、[协议](docs/list_weighting.md) | [Slurm 8651](runs/list_weighting_8651/report.md) |
| 排名分段聚合与随机分组 | [list_segmented.py](scripts/list_segmented.py)、[实验说明](docs/list_segmented_experiment.md) | [Slurm 8703](runs/list_segmented_8703/report.md) |
| 源列表共现聚类与匹配组大小对照 | [list_cluster.py](scripts/list_cluster.py)、[实验说明](docs/list_cluster_experiment.md) | [Slurm 8708](runs/list_cluster_8708/report.md) |
| 共现聚类簇数搜索 | [list_cluster_search.py](scripts/list_cluster_search.py)、[搜索规则](docs/list_cluster_count_search.md) | [Slurm 8711](runs/list_cluster_search_8711/report.md) |
| 为什么源域信息有效 | [诊断分析](docs/cdr_effectiveness_analysis.md) | [汇总 CSV](docs/cdr_effectiveness_results.csv) |
| 论文/代码参数核对 | [参数审计](docs/pfcdr_paper_configuration_audit.md) | [审计 JSON](results/paper_config_audit_20260922/audit.json) |
| 研究定位与相关工作 | [定位及文献核查](docs/list_research_positioning_and_novelty_audit.md) | 候选贡献分析，不代表新颖性保证 |
| 优化故事线设计记录 | [路线设计](docs/list_remaining_optimization_tracks.md) | IB、信息采集、Engram启发组合记忆；细粒度聚合暂停 |
| IB/记忆原型与固定K对照，8版本×2档×3种子；R2核心尚未实现 | [实际实施与分析](docs/list_three_story_experiments.md)、[方案核对](docs/list_implementation_alignment_audit_20260925.md) | [48模型汇总](runs/list_three_stories_8760_8763/report.md)、[比较表CSV](runs/list_three_stories_8760_8763/comparison.csv) |
| 条件高斯、20/60步条件能量、按用户批处理对照 | [模型](scripts/list_ambiguity_model.py)、[实验说明](docs/list_ambiguity_story_experiment.md) | [Slurm 8913](runs/list_ambiguity_8913/report.md) |
| 高斯和20步条件能量在另外两个任务的结果 | [入口](scripts/list_ambiguity_transfer.py)、[实验说明](docs/list_ambiguity_transfer_experiment.md) | [Slurm 8923](runs/list_ambiguity_transfer_8923/report.md) |
| 高斯评分噪声方差扫描 | [入口](scripts/list_gaussian_noise.py)、[结论](docs/list_gaussian_noise_experiment.md) | [Slurm 8952](runs/list_gaussian_noise_8952/report.md) |
| 高斯初始std、下限、上限顺序搜索 | [入口](scripts/list_gaussian_sequential.py)、[结论及后续固定策略](docs/list_gaussian_sequential_experiment.md) | [Slurm 8963](runs/list_gaussian_sequential_8963/report.md) |
| 朗之万六组参数独立扫描，训练/预测步长分开 | [模型](scripts/list_langevin_sensitivity_model.py)、[入口](scripts/list_langevin_sensitivity.py)、[结论](docs/list_langevin_sensitivity_experiment.md) | [90配置/126行结果](runs/list_langevin_sensitivity_8968/results.csv)、[复核](runs/list_langevin_sensitivity_8968/verification.json) |
| 全部优化的同数据、同种子汇总 | [CPU汇总脚本](scripts/summarize_list_optimization_overview.py)、[总览文档](docs/list_optimization_overview.md) | [逐种子CSV](docs/list_optimization_overview_results.csv)、[汇总核对](docs/list_optimization_overview_audit.json) |

源/目标重复物品已在新版任务中从源侧删除。早期未清洗的 Movie→Music 结果仅作历史记录，
不能与当前去重 Music→Movie 混比。历史入口见 [experiment_progress.md](docs/experiment_progress.md)。

## 数据、环境与复现范围

数据来自 Amazon Reviews 2014 5-core 的 Books、Movies_and_TV、CDs_and_Vinyl。
[下载脚本](scripts/download_data.py)已包含三个域；固定上游版本见
[upstream_repositories.json](upstream_repositories.json)，PFCDR 固定为
`c88e6fe7bd52ea258e686d5850cf5422d2b35a44`。

```bash
python -m pip install -r requirements.txt
python scripts/setup_workspace.py
python scripts/download_data.py
```

以上命令准备依赖、上游和原始数据，不会复现整张结果表。
历史训练脚本按两卡 Slurm 编排，依赖原始数据生成的划分、源权重和查询缓存；
其中服务器路径和前置运行号固定，需要在新环境中配置。
请先读 [当前版本的复现说明](docs/current_list_version.md)，不要向已完成的运行目录重复写入。

仅重建优化总览表时，无需数据集、GPU或第三方Python包：在完整仓库中运行
`python scripts/summarize_list_optimization_overview.py`，读取已归档的结果CSV及协议，
核对三种子均值、同种子基线和数据清单哈希，更新总览文档与配套CSV。
它不会训练模型或重算逐样本预测。模型级验证脚本仍需要服务器保留的数据、查询和checkpoint。

GitHub 保存代码、参数、训练指标、选模记录、汇总表和审计结果；不上传原始/处理数据、
ID 映射、完整查询列表、逐样本预测、模型 checkpoint、完整运行日志或第三方 checkout。
历史 JSON 的绝对路径和文件哈希保留用于溯源；被排除的大文件需要本地保留或重新生成。

排名分段和源列表共现聚类已作为独立实验实现；结果及限制见对应实验说明。目标相关的簇选择和列表邻居等仍未实现，不属于当前主版本。
