# LAPT：实现、参数与运行说明

LAPT 是 **Limited-Access Preference Transfer（有限访问下的偏好迁移）**。
当前版本从源平台接收每个用户的 Top-100 物品 ID 及其顺序，在目标平台训练评分预测模型。
模型不接收源评分、返回分数、源用户历史或源模型参数，也不使用 SID 或粗码。
实验中的源模型由评测端训练并冻结，随后生成列表；这与目标端获得源模型访问权限是两回事。

论文叙事见 [outline](lapt_paper_outline.md)。两个固定挑战为：

- **共享模式与用户差异混杂**：列表共同内容与用户差别混在一起，直接聚合可能丢失差别。
- **跨域迁移歧义**：相似的源域列表仍可能对应不同的目标域偏好。

## 当前工作流

输入为 `[用户数, 100]` 的有序整数 ID，0 只用于缺失列表的填充。
原 ID 可以做一一对应的连续编号；这不会把多个物品合成一个粗码。
目标端为这些源 ID 训练独立的 32 维向量表。

**DPE（Distinctive Preference Enhancement，差异化偏好增强）** 对应
[list_shared_reconstruction_model.py](../scripts/list_shared_reconstruction_model.py)：

1. 用物品向量、可训练的排名编码和原始排名先验，形成四个用户条件的模式向量。
   四个查询参数跨用户共享，但输出随每个用户的列表变化。
2. 用排名位置作为查询，从四个模式向量重构每个位置的物品表示；解码器没有直接读取该位置 ID 的旁路。
3. 逐位置计算“原物品向量减重构向量”，分别聚合重构部分和残差部分，再映射相加，得到列表表示。
4. 训练时均匀遮挡 20% 的列表位置，通过剩余内容预测被遮住的 ID。
   只使用当前阶段训练用户的列表及其 ID 集合，不把验证、测试用户用于辅助训练。

两路聚合都保留排名信息。这里没有频次统计或按频次降权，也没有保证某一层只学热门物品、
另一层只学冷门物品。新模块增加 11,584 个可训练参数，并增加辅助损失。

**CEPS（Conditional Energy-based Preference Sampling，条件能量驱动的偏好采样）** 沿用
[条件能量模型](../scripts/list_ambiguity_model.py)、
[训练和预测步长控制](../scripts/list_langevin_sensitivity_model.py)及
[固定采样配置](../scripts/list_coarse_sid_energy_model.py)。
列表表示经 MLP 生成高斯基础分布的均值与方差，条件能量网络进一步改变潜在偏好分布。
训练时分别采样列表条件先验与加入目标训练评分后的后验；推理只根据源列表进行采样。
采样器是带接受／拒绝校正的郎之万方法（MALA），采样对象为目标域偏好向量，随后取样本平均。
它没有生成新的源物品列表，也不使用目标测试评分形成后验。

最终预测为目标物品的普通训练均值，加上采样用户向量与目标物品向量的内积修正。
验证集选择全局修正系数 λ；λ=0 时最终结果就是普通物品均值。
模型可以表达条件差异，但这些实验没有证明其不确定性已经得到校准。

部分公共工具保留 `coarse_sid`、`raw_fixed_ablation` 等历史文件名，因为本轮直接复用了其训练循环、
采样器和初始化对齐逻辑。LAPT 路径读入原 ID；`coarse_vocab=256` 仅用于复现旧初始化的随机数消耗，
不构造或读取粗码。为避免缺少 import，本分支也收录这些公共工具的依赖。

## 固定参数与比较口径

[参数摘要 JSON](../configs/lapt_mf_seed2020.json)用于阅读核对；训练入口目前仍使用脚本中的配置，
不会自动加载这个摘要。参数与已完成实验一致，本次发布没有更改模型或训练算法。

| 项目 | 当前值 |
| --- | --- |
| 源接口 | 有序 Top-100 原物品 ID；无分数 |
| LAPT 表示／MLP 隐藏维度／dropout | 32／64／0.1 |
| DPE 模式数／遮挡比例／辅助系数 | 4／0.2／0.1 |
| 辅助损失 | 遮挡 ID 平均交叉熵，除以 log(训练 ID 集合大小) |
| LAPT 和 LIST V1＋郎之万训练批次 | 64 个完整重叠用户及其训练评分 |
| 原 LIST V1 训练批次 | 2048 条交互，包含目标训练集全部用户 |
| 训练预算／优化器 | 10 轮／Adam，lr=0.001，weight decay=1e-5 |
| 潜变量初始标准差／范围 | 0.5／[0.05, 1] |
| 评分噪声方差 | 1 |
| MALA 训练／预测步数 | 20／20 |
| MALA 训练／预测步长 | 0.25／0.25 |
| MALA 训练／预测链数 | 32／128 |
| 能量界 | 5 |
| λ 候选 | 0、0.25、0.5、0.75、1 |
| 选择方式 | 验证 RMSE 选择轮次与 λ，再重新初始化并在完整外层训练集重训 |

新增 DPE 的对照为相同批次、采样器和旧参数初始化的 LIST V1＋郎之万。
原 LIST V1 的批次单位不同，不能将它与 LAPT 的全部差异解释为一个模块的效果。

## 结果与代码入口

| 实验 | 范围 | 代码 | 结果 |
| --- | --- | --- | --- |
| MF 全量比较 | 3 任务 × β20/50/80 × 6 方法，seed2020 | [训练](../scripts/lapt_mf_full.py)、[两卡 Slurm](../scripts/slurm_lapt_mf_full.sh) | [分析与表格](lapt_mf_full_seed2020.md)、[PDF](lapt_mf_full_seed2020.pdf)、[LaTeX](lapt_mf_full_seed2020.tex)、[仅表格](lapt_mf_full_seed2020_tables.tex)、[CSV](lapt_mf_full_seed2020_results.csv) |
| 最初的 DPE 验证 | DNN，3 任务 × β20/80，seed2021 | [训练](../scripts/list_shared_reconstruction_study.py)、[两卡直接运行](../scripts/run_list_shared_reconstruction_direct.sh) | [报告](list_shared_reconstruction_seed2021_report.txt)、[CSV](list_shared_reconstruction_seed2021_results.csv)、[实现记录](list_shared_reconstruction_seed2021_implementation.txt) |

MF 六种方法为 TGTOnly、EMCDR、PFCDR、LIST V1、LIST V1＋郎之万、LAPT。
三任务为 Book→Movie、Book→Music、去重 Music→Movie，重叠用户分别为 37,388、16,738、17,904。
β 表示重叠用户中的目标冷用户测试比例。

MF 结果中，LAPT 相对 LIST V1＋郎之万：MAE 改善 3/9、持平 2/9；RMSE 改善 4/9、持平 2/9。
三个 LAPT 设置选择 λ=0。DNN seed2021 的 MAE 改善 5/6、RMSE 改善 2/6，且六组均超过普通物品均值。
这些是不同 backbone、不同种子的独立实验，不能合并为均值或宣称稳定全面领先。

MF 使用 PFCDR 作者公开默认值，包括 10 维 MF、10 轮训练、300 次反演及 K=70,000。
源侧去重、独立验证选模、固定划分与单种子运行不同于原论文的完整寻优及十次平均。
TGTOnly、EMCDR、PFCDR 保留作者原始 MF 前向，不额外加入均值回退；LIST 系列使用均值残差。
完整说明及核验结果见上述 MF 报告。

## 环境与 CPU 自检

```bash
python -m pip install -r requirements.txt
python scripts/setup_workspace.py
CUDA_VISIBLE_DEVICES='' python scripts/selftest_list_shared_reconstruction.py
```

`setup_workspace.py` 获取固定提交的 PFCDR 上游代码。CPU 自检仅使用合成小张量，不下载数据、不训练正式模型。
它检查遮挡 ID 没有输入旁路、排名变化影响表示、训练用户边界、梯度、初始表示和采样对齐、保存恢复一致性。
正式数据仍由 [下载脚本](../scripts/download_data.py)准备，当前实验依赖 Amazon Reviews 2014 的三个 5-core 域。

## 在原实验服务器重跑

以下是**已有前置数据的服务器命令**，不是空仓库一键复现。使用新的运行目录，避免覆盖已完成结果。

```bash
study_root="$PWD/runs/lapt_mf_new_seed2020"
python scripts/lapt_mf_full.py prepare --root "$study_root"
sbatch scripts/slurm_lapt_mf_full.sh "$study_root"
# Slurm 完成后：
python scripts/verify_lapt_mf_full.py --root "$study_root"
python scripts/finalize_lapt_mf_pdf.py --root "$study_root"
```

MF 入口依赖 `runs/paper_list_dnn_8638` 的数据划分和 `runs/small_task_comparison_8650/protocol.json`。
路径中的 DNN 是历史目录名；本轮只复用数据划分，源模型与 Top-100 重新用 MF 生成。
数据准备逻辑见 [paper_list_dnn.py](../scripts/paper_list_dnn.py)；完整 CSV、用户划分数组、查询与检查点没有上传。
换服务器时要先重建相同前置数据、配置与哈希，并调整这些路径，不能直接使用缺失缓存的训练命令。
DNN 归档入口还依赖前序原 ID 消融和频次组对照的运行产物，具体依赖写在脚本中。

Slurm 脚本包含本服务器的工作路径、Python 环境、分区，以及 GRES 设备号到 UUID 的映射，
其他集群须按其实际配置调整。直接运行入口也保留当时的执行记录校验；不用于绕过集群调度要求。

MF 导出分两步：[export_lapt_mf_full.py](../scripts/export_lapt_mf_full.py)生成表格，
[finalize_lapt_mf_pdf.py](../scripts/finalize_lapt_mf_pdf.py)修正冻结导出器的排版并核对 108 个指标。
报告工具还依赖 Tectonic 和 PyMuPDF（本次为 0.15.0 和 1.28.2），路径在脚本中固定。
如果只编译已经发布的最终 `.tex`，可使用常规 LaTeX 工具链，无需训练数据。

GitHub 保存代码、参数、论文 outline、汇总指标和结果核验摘要。
原始数据、用户或物品映射、完整列表、逐样本预测、检查点和第三方 checkout 保留在本地。
