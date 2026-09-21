# 第一轮复现记录

> 历史记录。后续三任务、EMCDR、计时和数据污染发现以
> [当前进度与分析](experiment_progress.md) 为准。

## 环境与版本

- 使用现有 `/root/miniconda3/bin/python`，不修改 base 环境的依赖。
- Python 3.12.3，PyTorch 2.11.0+cu130，pandas 3.0.2，numpy 1.26.4，scipy 1.12.0。
- PFCDR：https://github.com/walcheng/PFCDR ，commit `c88e6fe7bd52ea258e686d5850cf5422d2b35a44`。
- DBGRME：https://github.com/Vencent-Won/DBGRME ，commit `d0503d58abbe88962a052ade0579189b1d8758e4`。
- 数据按 PFCDR README 下载 Stanford 的 Amazon 2014 Books / Movies_and_TV 5-core 文件，无子采样。
- 下载归档 SHA256：Books `d22b38a7066e2ac3b25ffb66bc1264d8974ebe8f9dc51ed2e4b0f6ed43358a65`；Movies_and_TV `2219c5d688f8f923884626054ccd9098b9fe203228a7fe97d621749d5f26cf6e`。完整 gzip 解压通过 CRC 校验。

## 协议与必要适配

- 固定 `PYTHONHASHSEED=2020` 与随机种子2020，保存官方映射过程的 ID 表。
- 数据划分调用官方 `mapper/split/save`。Python 3.12 不支持 `random.sample(set, k)`，只将该参数转换为 list，保留原迭代顺序。
- 只准备任务1、比例[0.8,0.2]。官方 main 默认实际是任务3、GPU1，预处理还遍历不存在的任务4，故使用显式入口。
- MF 维度10；源域与目标域训练10轮，Adam，lr=.01，weight_decay=0；跨域原型映射10轮，lr=.005。
- 用批量 GPU 索引代替逐条 TensorDataset 索引；保留 DataLoader 的 sampler、随机顺序、batch size 和每批优化步数，已逐批检查顺序相同。
- 模型反演调用官方 Inversion，300次迭代，batch=200，lr=1e-4，weight_decay=1e-4，目标分数5。
- 反演的 item weighted sum 用矩阵乘法替代广播乘法后求和，避免分配 B×I×D 中间张量。已检查前向和输入梯度等价。
- MF 投票实现实际硬编码 topk=70000，保留该行为，而非 config.json 中的85000。
- 同一个冷启动用户的投票均值向量按用户缓存；利用内积线性性计算每个物品评分。运行时与官方预测做数值核对。
- 跳过不在本轮实验范围内的 EMCDR 映射训练；源模型训练仍调用官方训练函数。WB与BB共用源 victim、目标模型、测试集和初始 prototype mapping。
- 每轮记录测试 MAE，主结果预先固定第10轮，不根据测试标签选模型；官方最低测试 MAE 可另行从日志读取。
- 官方目标侧投票候选包含全部目标用户（包括未训练的 cold 用户向量），首轮保持该行为。
- 官方 TGT 的完全冷启动用户向量未经训练，因此其误差可能很高。原样报告，解释跨域收益时明确这一限制。

## DBGRME 接口不兼容与黑盒组假设

DBGRME 官方 `DBGRME.py:get_synthesis_data` 将虚拟用户及合成交互接入 victim 图，`V.init_fakers()` 初始化虚拟用户；其 surrogate 测试也接收真实训练图。不能将这套调用直接当作本文档中已有用户 ID → Top-K 的 MF API。

首轮采用普通 Top-K＋分数 MSE 蒸馏作为朴素 extraction baseline，**不是 DBGRME 复现**，也不是新方法。用户已明确选择这一替代基线。

- 固定每个重叠用户查询一次，Top-100，返回物品 ID 和 victim 原始预测分数，不裁剪、不使用源日志过滤 seen items。
- 查询包括冷启动目标用户：他们是源域已有用户，这是允许的源 API 信息，不读取其目标域测试标签。
- source-service 导出 `queries/topk_scores.npz`，learner 只读 API 输出和公开 ID 维度，随机初始化相同维度的 MF。
- learner 使用 Linux Landlock 文件访问白名单；原始交互、victim/base checkpoint 不可读取，启动时显式测试拒绝访问。
- BB 下游只读 target checkpoint、初始映射和 surrogate；不加载含 victim 的 base checkpoint。
- 分数蒸馏固定10轮，Adam lr=.01，batch=512，不以目标域测试误差调整 extraction 超参数。
- 本轮差距只能评价这个指定 API 和提取基线，不能证明黑盒访问本身的不可消除损失。

## 入口与产物

`scripts/download_data.py` → `scripts/prepare_data.py` → `scripts/run_whitebox.py` → `scripts/run_blackbox.py`。

各步骤日志在 `logs/`，配置在 `configs/`，模型和原型在 `checkpoints/`，逐样本预测及最终汇总在 `results/`。
