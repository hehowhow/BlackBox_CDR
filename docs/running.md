# 运行说明

## 准备空目录

仓库保留历史小型结果，部分脚本拒绝覆盖已有结果。推荐：

```bash
python scripts/setup_workspace.py --fresh-workspace /absolute/path/to/new-run
cd /absolute/path/to/new-run
```

这个命令复制代码、配置、文档和依赖声明，创建必要的空运行目录，并 checkout
固定 commit 的 PFCDR。它不复制历史结果、数据或checkpoint，并拒绝已有目标目录。
`--skip-upstream` 可离线只准备目录；`--with-dbgrme` 可选下载接口研究用仓库。
不传 `--fresh-workspace` 则仅准备当前目录，保留原有内容。

## 历史实验

按 README 命令依次运行。`run_extra_tasks.py` 依赖先准备 Book/Movie 中间数据，
自动下载和处理 Music，随后分别运行源/目标预训练、API、List、蒸馏、WB/BB反演和
映射。阶段日志保存在对应 `runs/<task>/logs`，失败后从日志定位；完成标志只用于
同一次额外任务的恢复。

`run_emcdr.py --root /absolute/path/to/task` 要求此任务已有 WB、BB、List 的预测
及 `base_seed2020.pt`。它验证未训练的映射初始化，复用两域MF，只训练官方EMCDR
映射，不复训源/目标MF。此时耗时不能当作完整从零训练耗时。

## 从 raw 独立计时

先确保 Movie/Music raw gzip 已下载；可在完成额外任务后运行：

```bash
PYTHONHASHSEED=2021 python scripts/benchmark_fresh_raw.py
```

它拒绝已有 `runs/fresh_movie_music_seed2021`，两条路线分别重新解析raw、划分、
初始化及训练。WB使用两域MF和反演；List仅训练源服务MF，再生成新查询列表并训练
新学生。不会重用历史模型或中间数据。源域shuffle随机流单独固定，以核对两次
独立源训练是否一致。完整墙钟包括解析到结果保存，不包括下载；OS页缓存不清空。
现有数据协议存在重复评论，本入口仍复现该历史协议。

## 审计

完成三组和fresh raw实验后，可执行：

```bash
python scripts/audit_list_pipeline.py
python scripts/audit_cross_domain_duplicates.py
python scripts/audit_raw_review_duplicates.py
python scripts/audit_list_replay.py
```

审计需要本地原始/处理数据、模型和预测，这些大文件不在Git中。
`audit_list_replay.py` 禁止加载模型，记录输入文件读取，只有学生训练结束后才能读
测试标签，并比较真实/打乱测试标签的两次确定性新训练。其查询输入仍来自被审计
源模型，因此不能解决上游的数据污染。诊断目录 `results/list_audit/replay_*`
不上传；汇总JSON保留历史路径方便实验机器溯源。

## 已知运行边界

- wrapper只验证过本次Linux/CUDA环境，尚未提供CPU执行路径。
- 普通蒸馏依赖Landlock，`extract_surrogate.py` 的原依赖白名单含
  `/root/miniconda3`。在不同conda安装路径上应调整白名单，不能简单去掉隔离。
- 固定数据划分依赖PYTHONHASHSEED和Python集合迭代；更换环境可能无法逐行复现旧ID。
- 普通List训练未强制确定性CUDA操作；测试标签扰动诊断开启了确定性模式，所以
  诊断模型数值可与历史普通训练略有差异。
- 原脚本使用完整数据、保存必要中间产物，注意可用磁盘；本仓库没有清洗数据修复。
- 学习器使用源输出不等于获得形式化隐私保证。
