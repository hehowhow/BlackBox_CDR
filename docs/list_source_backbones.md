# LIST：只更换源域 Top-K 模型

已完成 Slurm 8596，COMPLETED / 0:0，全部核验 PASS。
结果见 [实验结论](../runs/list_source_backbones_8596/conclusions.md) 和
[数值表](../runs/list_source_backbones_8596/comparison.csv)。
同一 LIST 使用 DNN 列表得到 MAE=0.812435、RMSE=1.067772，
均优于普通物品均值（0.834888 / 1.091984）和平滑均值（0.842519 / 1.086142）；
源 MF 和 GMF 列表对应 LIST 仍未超过均值。

本轮复用 `runs/pfcdr_backbones_8589` 的 MF、GMF、DNN 源模型，为 Book→Movie
全部 37388 个重叠用户分别生成 Top-100。源模型相同数据、维度 10、10 epoch、种子 2020。
源 GMF 使用其真实线性加权评分，DNN 使用其真实用户线性变换评分。
不使用 PFCDR 匹配公式、目标模型或反演原型生成推荐列表。

API 保持原规则：完整源物品目录、不排除已交互物品、不裁剪评分。
每个 backbone 保存 `queries/<name>/topk_ids.npz`（user_ids、item_ids），
另存仅供导出审计的 `scores.npy`。对四个用户用原生 forward 完整评分整个目录复核。

下游 LIST 保持原架构：ID embedding 32、排名权重池化、32→64→32 的 ReLU 编码器、
dropout 0.1、目标物品内积、训练全局均值和物品偏置。
学生只读取 Top-100 ID，不读取教师权重、源交互、原型或返回分数。
三种列表共用返回 ID 的并集词表，仅用于对齐 ID 与随机初始化；不读取额外目录信息。
每组从同一个学生 checkpoint 开始，共用训练批次顺序和 dropout 随机种子。

数据沿用 8589 的固定 60/20/20 重叠用户划分。所有目标独有用户保留训练，
其源列表为空，只通过全局均值和物品偏置预测。
学生训练 10 epoch，Adam lr=0.001、weight decay=1e-5、batch=2048。
每组以验证 RMSE 选 epoch 1–10，三组全部冻结并核验后统一评价测试集。
此处与历史 LIST 的固定最后 epoch 报告不同，但三组使用完全相同的验证规则。
训练未出现的目标物品按已有协议回退平滑均值；主指标 RMSE，同时报告 MAE。

学生实现与原版类进行前向、梯度和空列表用户行为自检。
冻结代码、数据、源 checkpoint、查询、初始权重、训练顺序和选模结果哈希；
核验验证最小值以及 checkpoint 恢复后的完整验证/测试预测。
报告三种源列表的交集比例、同一学生下的效果，以及相对 MF 列表的配对用户簇区间。

范围：Book→Movie、单种子；没有改变学生评分 backbone，没有搜索学习率或 K。
历史测试集已在此前实验使用，本轮属于探索性比较，无法单独证明其他数据集效果。

入口：`sbatch scripts/slurm_list_source_backbones.sh`。申请 1 张 GPU、4 CPU、24G 内存，
顺序执行，产物写入独立 `runs/list_source_backbones_<jobid>`，不覆盖历史结果。
