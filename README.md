# BlackBox CDR — 实验代码与审计记录

研究源域仅提供 Top-K 推荐输出时的目标域冷启动评分预测。使用 Amazon Reviews
2014 5-core，已完成 Book→Movie、Movie→Book、Movie→Music，以及一次独立从 raw
开始的 Movie→Music 计时复跑。

**当前结果不能作为方法优越性的结论：** Movie/Music 有跨类别重复评论，seed=2020
目标测试评分中 5,577 条（5.96%）已出现在源域训练集；seed=2021 为 5,236 条
（5.81%）。此外，List 在三个任务上均未超过只用目标域训练评分的物品均值基线。
尚未清洗重复记录并重新训练。

请先阅读 [实验进度与分析](docs/experiment_progress.md) 和
[完整审计报告](results/list_audit/report.md)。

## 已完成的实验

seed=2020，20% 重叠用户作为目标域冷启动用户，固定最后一轮。

| 任务 | EMCDR MAE ↓ | PFCDR-WB | 蒸馏＋PFCDR | List | 目标域物品均值 |
|---|---:|---:|---:|---:|---:|
| Book→Movie | 1.1761 | 0.9679 | 1.0499 | 0.8581 | **0.8334** |
| Movie→Book | 1.1193 | 1.0771 | 1.2790 | 0.7991 | **0.7874** |
| Movie→Music（受污染） | 1.6534 | 1.6392 | 1.6494 | 0.7614 | **0.7420** |

- PFCDR-WB：白盒模型 → 原型反演 → 跨域映射。
- 蒸馏＋PFCDR：Top-100＋分数 → MF surrogate → PFCDR；不是 DBGRME 复现。
- List：Top-100 ID＋排名加权池化 → MLP → 目标域评分。
- EMCDR：官方 MF 用户 embedding 线性映射，直接读取真实源域 embedding。
- 物品均值：不用源域信息、不训练模型，按目标域训练集的物品平均评分预测。

## 仓库内容

`scripts/` 是下载、预处理、训练、评估和审计脚本；`configs/` 保存固定配置；
`docs/` 保存原始想法、历史计划、复现说明和当前分析；`results/` 与 `runs/`
保存小型汇总、训练曲线、协议和审计结果。

不上传原始/处理数据、ID 映射、查询缓存、逐样本预测、checkpoint、完整运行日志、
第三方 checkout 或 Python 缓存。JSON 中的历史本机路径仅用于溯源，对应大文件不
在 Git 中。详细结果索引见 [进度报告](docs/experiment_progress.md)。

## 复现

原环境：Linux、Python 3.12.3、PyTorch 2.11.0+cu130、CUDA GPU；依赖见
[requirements.txt](requirements.txt)。按本机驱动安装可用的 CUDA PyTorch。

建议创建独立的空产物目录，避免覆盖本仓库保留的历史结果：

```bash
python -m pip install -r requirements.txt
python scripts/setup_workspace.py --fresh-workspace /path/to/new-run
cd /path/to/new-run

python scripts/download_data.py
PYTHONHASHSEED=2020 python scripts/prepare_data.py
python scripts/run_whitebox.py
python scripts/run_blackbox.py
python scripts/run_list_encoder.py
python scripts/run_emcdr.py --root "$PWD"

PYTHONHASHSEED=2020 python scripts/run_extra_tasks.py
python scripts/run_emcdr.py --root "$PWD/runs/movie_to_book"
python scripts/run_emcdr.py --root "$PWD/runs/movie_to_music"
```

这些入口复现的是当前记录的历史协议，**包括未清理的 Movie/Music 重叠**，不是
修正后的正式评估。下载和训练需要数 GB 空间，勿并发运行同一输出目录。
普通蒸馏隔离还需要 Linux Landlock；其原白名单含 `/root/miniconda3`，迁移环境
需要调整。更多入口、约束和计时口径见 [运行说明](docs/running.md)。

## 上游代码

PFCDR 使用 [walcheng/PFCDR](https://github.com/walcheng/PFCDR) 固定版本；wrapper
保留原算法并加入经过数值核验的计算优化。DBGRME 仅用于接口调查，可选下载。
版本见 [upstream_repositories.json](upstream_repositories.json)，由 setup 脚本
独立获取。本仓库不打包第三方源码；使用条件以各上游仓库为准。
