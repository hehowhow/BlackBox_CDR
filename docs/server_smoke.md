# 本服务器 Slurm 验证

仓库位置：`/home/hehao32/BlackBox_CDR`。
环境：管理员预装的 `/home/hehao32/miniconda3` 下的 `blackbox-cdr`，
Python 3.12，依赖按根目录 `requirements.txt` 安装。
遵循 `/home/hehao32/serverRule.md`，计算任务通过 Slurm 提交。

## 2026-09-22 验证结果

**PASS**：GPU 作业 `8553` 为 `COMPLETED`，退出码 `0:0`，耗时 `00:05:50`。
实际分配物理 GPU 7：NVIDIA RTX PRO 6000 Blackwell Server Edition，
计算能力 12.0，PyTorch `2.11.0+cu130` / CUDA 13.0。
原始数据解析作业 `8551` 和划分/校验作业 `8552` 也均成功结束。

完整 Book→Movie 数据：8,898,041 条源域交互，1,541,404 条目标域训练交互；
37,388 名重叠用户中，29,910 名用于映射训练，7,478 名冷启动用户用于测试，
测试评分共 156,129 条。三组反演原型分别覆盖 29,910 / 7,478 / 29,910 名用户。

| 验证指标 | 数值 |
| --- | ---: |
| PFCDR-WB MAE | 3.558627 |
| PFCDR-WB RMSE | 3.813257 |
| 预训练/映射 epoch | 1 |
| 反演迭代次数 | 2 |

这是低轮数的功能验证结果，不能作为模型收敛效果或论文复现指标。
反演前向/梯度一致性、官方投票预测一致性、预测数值和测试覆盖检查均通过。

结果目录：`runs/slurm_smoke_8553/`；简明结果为其中的 `smoke_summary.json`，
各阶段日志见 `logs/`，模型见 `checkpoints/`，逐样本预测与训练指标见 `results/`。
外层 `logs/slurm_accounting.txt` 保存 Slurm 完成记录，
`logs/environment.freeze.txt` 保存实际依赖版本。

## 数据准备

原始数据来自 [PFCDR README](https://github.com/walcheng/PFCDR#dataset)
指定的 [Stanford Amazon Reviews 2014](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/)
5-core 数据集，共三个完整 gzip：

| 类别 | 交互条数 | gzip 字节数 |
| --- | ---: | ---: |
| Books | 8,898,041 | 3,223,678,899 |
| Movies_and_TV | 1,697,533 | 708,988,936 |
| CDs_and_Vinyl | 1,097,592 | 490,831,332 |

原始数据放在 `data/raw/`，解析后的三列 CSV 放在 `data/mid/`。
Book→Movie 的完整训练/测试划分在
`data/ready/_8_2/tgt_Movies_and_TV_src_Books/`。
`data/id_maps/` 保存 ID 映射，`data/split_manifest.json` 保存划分统计，
`data/download_manifest.json` 保存来源、字节数、完整解析后的条数及本地 SHA256。
SHA256 是本地复核值；上游未提供用于对照的官方 SHA256。

```bash
cd /home/hehao32/BlackBox_CDR
source /home/hehao32/miniconda3/etc/profile.d/conda.sh
conda activate blackbox-cdr
python scripts/setup_workspace.py
python scripts/download_data.py
sbatch scripts/slurm_prepare.sh
```

预处理任务成功结束后运行下面的 GPU 验证。也可用
`sbatch --dependency=afterok:<预处理作业ID> scripts/slurm_smoke.sh` 自动衔接。

## GPU 验证

```bash
cd /home/hehao32/BlackBox_CDR
sbatch scripts/slurm_smoke.sh
squeue -u hehao32
sacct -j <作业ID> --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES
```

脚本申请 1 张 GPU、4 CPU、32 GB 内存，时限 1 小时，使用 Slurm 设置的
`CUDA_VISIBLE_DEVICES`。不能只看 `nvidia-smi` 的利用率选卡：空闲卡可能已经
被其他作业预留，也不能用 wrapper 里写死的物理 GPU 0 覆盖调度结果。

运行已有 `scripts/experiment.py` 的 `train`、`invert`、`bridge` 三阶段，
使用完整 Book→Movie 数据与官方模型。源/目标 MF 预训练和跨域映射均为
1 epoch；反演为 2 次迭代，其余沿用仓库配置。此设置只验证程序可运行，
不能用于论文效果比较。

每次作业新建 `runs/slurm_smoke_<作业ID>/`，其中 `data` 链接到共享数据，
配置、模型、预测、日志与指标独立保存。历史 `results/` 和已记录的实验不被覆盖。

- `logs/smoke_<作业ID>.out/.err`：Slurm 输出。
- `runs/slurm_smoke_<作业ID>/logs/wb_{train,invert,bridge}.log`：各阶段详细日志。
- `runs/slurm_smoke_<作业ID>/environment.json`：GPU、PyTorch、配置和仓库版本。
- `runs/slurm_smoke_<作业ID>/smoke_summary.json`：最终验证结果。

成功条件：Slurm `COMPLETED` / `0:0`，输出 `SMOKE_TEST_PASS`；两个域训练、
全部重叠用户的原型反演和映射评估均完成，预测覆盖完整测试集，数值有限，
从逐样本预测重算的 MAE/RMSE 与训练日志一致。程序还校验优化后的反演计算、
输入梯度和投票预测与官方实现的一致性。

## 本次适配

- 下载入口补充 Music，与仓库三个域的数据需求一致。
- `experiment.py` 将 `inversion_*` 设置传入官方反演模块；原先这些配置被忽略，
  导致缩短迭代次数不生效。默认配置仍使用原来的 300 次迭代。
- 新增两个 Slurm 脚本；上游 PFCDR checkout 保持固定版本且未修改。
