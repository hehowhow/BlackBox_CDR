# DNN 源列表在其他数据集上的复核

已完成 Slurm 8603：两卡并行，9 分 30 秒，12 个源模型及 12 次 LIST 训练，全部核验 PASS。
结果见 [实验结论](../runs/list_source_transfer_8603/conclusions.md) 和
[三种子汇总](../runs/list_source_transfer_8603/comparison.csv)。
DNN 源列表在两个任务的全部六组配对种子上均优于 MF 列表的 MAE/RMSE，
但两个任务的 RMSE 均仍落后于平滑物品均值。
Movie→Book 的 DNN LIST 均值为 0.793978 / 1.023594；
Movie→Music（清洗后）为 0.751473 / 1.008283（MAE / RMSE）。

范围：Movie→Book、Movie→Music（清洗后）。对比 MF 和作者式 DNN 源模型生成的 Top-100，
目标端固定为原内积 LIST，不改变目标评分头。

每任务运行配对种子 2020/2021/2022。每个种子都重新训练 MF/DNN 源模型及各自 LIST，
共 12 个源模型、12 次学生训练。三个种子同时改变源与学生随机性；不是 3×3 全因子实验。

源模型沿用上一轮设置：维度 10、10 epoch、Adam lr=0.01、weight decay=0、batch=512。
每组 MF/DNN 对齐源用户/物品初始向量和批次顺序；DNN 仿射层按作者默认规则初始化。
源预训练固定最后一轮，不使用目标标签；生成完整目录 Top-100，不过滤源域已交互物品，
不裁剪预测分数。导出时用作者原生 forward 对四名用户的完整目录核对评分和 Top-100。

LIST 直接复用已核验的原训练实现：排名池化、32→64→32 ReLU 编码器、dropout 0.1、
目标物品内积、训练均值及物品偏置。每组 MF/DNN 共用返回 ID 的并集词表和初始学生权重，
相同训练批次顺序和 dropout 种子。学生只读 Top-100 ID，不能读取源数据、权重或返回分数。

学生训练 10 epoch，Adam lr=0.001、weight decay=1e-5、batch=2048，
以验证 RMSE 选 epoch 1–10。保留目标独有训练用户，无源列表时只走均值加物品偏置。
未见目标物品回退已有平滑均值；报告 MAE、RMSE、三次均值/样本标准差与配对用户簇区间。

复用 `runs/residual_study_20260922` 的固定 60/20/20 用户划分和训练集均值基线。
Movie→Music 从源域删除的 2441 个共享物品、35333 条评分保持删除；核对数据哈希、
源目标原始 ID 交集为零、删除列表中的物品不在清洗后的源目录。
本轮没有重做 5-core、时间过滤或数据划分。

全部 12 次源/学生流程完成并验证 checkpoint 恢复预测后，才统一读取目标测试标签。
没有调学习率、K 或训练预算；没有新增打乱列表消融。历史测试集已使用，本轮为探索性复核。
三个种子不能代表其他数据划分不确定性；区间条件于已训练模型，未校正多重比较。

入口：`sbatch scripts/slurm_list_source_transfer.sh`。两张 Slurm GPU 两路并行，8 CPU、48G 内存。
产物独立写入 `runs/list_source_transfer_<jobid>`，不覆盖此前结果。
