# LIST 残差主版本快照

归档日期：2026-09-24（实验完成于2026-09-23）。版本名称：`list-residual-v1`。
本快照保存已验证的排名聚合版本；两种加权为对照，不包含尚未实现的分段聚合。

## 模型和固定参数

源端使用作者式 DNN 生成每个用户的 Top100；目标学生只读取 ID/排名。
对源 ID 的学生向量按 `1/log2(rank+1)` 归一化加权求和，经 MLP 得到用户表示，
与目标物品向量内积，输出残差。最终预测为：

`prediction(u, i) = ordinary_training_item_mean(i) + lambda * residual(u, i)`

普通均值没有平滑。无源列表用户残差为零；目标训练未见物品回退训练全局均值。
编码器最后一层零初始化，从均值开始训练；额外物品偏置冻结为零。
源 Top100 不过滤已交互源物品。当前实验是固定源查询快照，不是线上动态 API 测试。

实现：[ResidualList](../scripts/small_task_comparison.py#L281)、
[共用 ListStudent](../scripts/list_source_backbones.py#L43)。
机器可读参数：[list_residual_v1.json](../configs/list_residual_v1.json)，
它是归档参数快照，不是现有脚本直接读取的 CLI 配置。

| 模块 | 参数 |
| --- | --- |
| 源骨干 | 作者 DNN：用户仿射变换后与物品向量内积；dim=10 |
| 源预训练 | Adam，lr=0.01，weight_decay=0，batch=512，10 epochs |
| 接口 | K=100；仅 ID 和排名；不使用返回分数、教师向量、源历史或文本 |
| 学生表示 | 源/目标 dim=32；MLP 32→64→32；ReLU；dropout=0.1 |
| 聚合 | 固定排名权重 `q_r=1/log2(r+1)`，列表内归一化 |
| 基底 | 目标当前训练集普通物品均值；未见物品使用训练全局均值 |
| 学生训练 | MSE；Adam，lr=0.001，weight_decay=1e-5，batch=2048 |
| 选模 | 最多10轮；验证 RMSE 联合选择 epoch=0..10 与 lambda∈{0,0.25,0.5,0.75,1} |
| 重训 | 相同初始化，在全部外层训练数据上训练选定轮数；lambda 固定 |
| 种子 | 模型种子2020/2021/2022；固定用户划分，split_seed=2020，validation_seed=20260923 |
| 任务 | 去重 Music→Movie；β20%/80%；外层训练重叠用户中20%用于内层验证 |
| 资源 | 两卡 Slurm；8650为8 CPU/48G，8651为8 CPU/32G |

Music→Movie 从源 Music 删除2,441个共享物品、34,881条评分；目标不变，
剩余17,904个重叠用户。测试用户无目标训练历史；保留目标独有用户参与物品均值统计。
作者 task3 的源 batch=256，本轮为保持已有 DNN 列表使用512；不是逐参数论文复现。

## 实际选中的参数和结果

| β | 模型种子 | 真实列表选定 epoch | lambda |
| --- | --- | ---: | ---: |
| 20% | 2020 | 4 | 0.75 |
| 20% | 2021 | 6 | 0.50 |
| 20% | 2022 | 5 | 0.50 |
| 80% | 2020 | 10 | 0.25 |
| 80% | 2021 | 10 | 0.25 |
| 80% | 2022 | 8 | 0.25 |

这些是验证集选出的结果，不是依据测试结果指定的超参数。

| 方法 | β20% MAE | β20% RMSE | β80% MAE | β80% RMSE |
| --- | ---: | ---: | ---: | ---: |
| 普通均值 | 0.828612 | 1.074618 | 0.840744 | 1.097641 |
| 原LIST | 0.822822 | 1.067575 | 0.870605 | 1.124753 |
| 当前LIST | 0.808497 | 1.056126 | 0.833717 | 1.089396 |
| 当前LIST错配列表 | 0.826968 | 1.073404 | 0.840470 | 1.097543 |

三种子指标均值；MAE/RMSE均越低越好。两种加权没有跨两档一致改善，不替换主版本。
β80%的EMCDR-DNN MAE更低，不能把LIST描述为所有指标最优。

- [8650完整报告](../runs/small_task_comparison_8650/report.md)、[逐种子指标](../runs/small_task_comparison_8650/per_seed.csv)
- [8651全方法报告](../runs/list_weighting_8651/report.md)、[九方法对比CSV](../runs/list_weighting_8651/main_comparison.csv)
- [8650独立核验](../runs/small_task_comparison_8650/independent_verification.json)、[8651独立核验](../runs/list_weighting_8651/independent_verification.json)

区间条件于当前模型和固定划分，未校正多重比较；历史测试曾用于探索。
新残差版尚未在其他方向完整验证；当前MAE/RMSE结果不是Top-N排序指标。

## 归档与复现依赖

保存原始实验脚本，不改写其历史代码哈希。依赖链为：

`Amazon raw → paper_list_dnn_8638 → small_task_comparison_8650 → list_weighting_8651`

1. [paper_list_dnn.py](../scripts/paper_list_dnn.py)负责去重、用户划分、源DNN与原LIST；
   两卡入口为[slurm_paper_list_dnn.sh](../scripts/slurm_paper_list_dnn.sh)。
2. [small_task_comparison.py](../scripts/small_task_comparison.py)复用第一阶段DNN权重、Top100和原LIST预测，
   训练当前残差版及EMCDR/PFCDR；入口为[slurm_small_task_comparison.sh](../scripts/slurm_small_task_comparison.sh)。
3. [list_weighting.py](../scripts/list_weighting.py)复用前两阶段材料运行加权对照；
   入口为[slurm_list_weighting.sh](../scripts/slurm_list_weighting.sh)。

现有入口包含本机工作目录、conda路径、LocalQ分区和前置运行号。
仅clone本仓库不能直接执行后两步：Git不包含它们依赖的checkpoint、数据、查询和逐样本预测。
在原服务器保留大文件时，可用新的输出目录启动重复实验；在全新机器上，需要先准备数据和固定PFCDR上游，
在独立工作副本中调整Slurm路径和前置运行常量，依次重新生成材料。
新的运行从prepare阶段生成自己的协议与哈希，不能改写历史哈希来冒充原实验。
不要向8650/8651等已经完成的运行目录重复写入。任何GPU训练仍须通过Slurm。

本次上传保留轻量的历史实验结果和诊断代码，以便追溯主版本的选择。
大文件及运行链接仍保留在本机；GitHub不承担数据/模型权重备份。
