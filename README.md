# BlackBox CDR — LIST 跨域冷用户评分预测

源推荐服务只提供 Top-K 物品 ID 和排名时，能否帮助预测另一个域的评分？
本仓库保存实验代码、固定参数、两卡 Slurm 入口、结果表与审计记录。

**当前主版本：DNN 源 Top100 → 排名加权聚合 → MLP → 普通目标物品均值上的残差修正。**
DNN 沿用 PFCDR 作者命名，实际为用户仿射变换后与物品向量内积。
学生不读取源评分、教师参数或返回分数；需要已知的跨域用户对应与重叠训练用户。

[当前版本与参数](docs/current_list_version.md) ·
[参数快照 JSON](configs/list_residual_v1.json) ·
[完整结果与分析](runs/list_weighting_8651/report.md) ·
[对比表 CSV](runs/list_weighting_8651/main_comparison.csv)

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

上述改进主版本目前只在去重 Music→Movie 两档完整验证；其他方向的旧 LIST 结果
不等于该新版本的效果。历史测试已用于探索，不是全新盲测。PFCDR 没有做 K/LR 搜索，
这些结果不是作者十次运行的逐参数复现，也不是各方法充分调参后的性能上限。

## 代码与结果入口

| 内容 | 实现/协议 | 结果 |
| --- | --- | --- |
| 三个论文方向、β20/50/80、DNN 原 LIST | [paper_list_dnn.py](scripts/paper_list_dnn.py)、[协议](docs/paper_list_dnn.md) | [Slurm 8638](runs/paper_list_dnn_8638/report.md) |
| 当前均值残差 LIST、错配对照及 EMCDR/PFCDR | [small_task_comparison.py](scripts/small_task_comparison.py)、[协议](docs/small_task_comparison.md) | [Slurm 8650](runs/small_task_comparison_8650/report.md) |
| 频次降权、目标相关加权 | [list_weighting.py](scripts/list_weighting.py)、[协议](docs/list_weighting.md) | [Slurm 8651](runs/list_weighting_8651/report.md) |
| 为什么源域信息有效 | [诊断分析](docs/cdr_effectiveness_analysis.md) | [汇总 CSV](docs/cdr_effectiveness_results.csv) |
| 论文/代码参数核对 | [参数审计](docs/pfcdr_paper_configuration_audit.md) | [审计 JSON](results/paper_config_audit_20260922/audit.json) |
| 研究定位与相关工作 | [定位及文献核查](docs/list_research_positioning_and_novelty_audit.md) | 候选贡献分析，不代表新颖性保证 |

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

GitHub 保存代码、参数、训练指标、选模记录、汇总表和审计结果；不上传原始/处理数据、
ID 映射、完整查询列表、逐样本预测、模型 checkpoint、完整运行日志或第三方 checkout。
历史 JSON 的绝对路径和文件哈希保留用于溯源；被排除的大文件需要本地保留或重新生成。

后续讨论的分段聚合、多表示、列表邻居等尚未实现，不属于当前主版本。
