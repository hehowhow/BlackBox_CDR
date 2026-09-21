# Baby Check：第一轮结果

> 历史结果：本报告中的增益基于随机冷用户向量的弱TGT基线。
> 后续目标物品均值优于List，Movie/Music还发现重复评论污染。
> 最新解释见 [实验进度](../docs/experiment_progress.md) 和 [审计报告](list_audit/report.md)。

Book→Movie；MF；β=20%；seed=2020；固定第10轮。

| Setting | MAE ↓ |
| --- | ---: |
| TGT | 4.178897 |
| PFCDR-WB | 0.967869 |
| PFCDR-BB（普通分数蒸馏） | 1.049850 |

G_WB=3.211028；G_BB=3.129047；BB−WB=0.081981；RecoveryRatio=0.9744689424426646.

Q1：本次实验支持正收益；配对用户 bootstrap 95% CI：[3.086399, 3.321234]。

Q2：本次黑盒基线明显劣于白盒；差距95% CI：[0.065328, 0.097544]。

Q3：本次实验支持正收益；95% CI：[3.006230, 3.238026]。

限制：只有一个划分与训练种子，区间不包含重训练不确定性。黑盒组不是 DBGRME，不能将其差距解释为所有黑盒方法的固有上限。

官方 TGT 对完全冷启动用户使用未训练的随机用户向量；因此相对该基线的收益不能单独证明源知识优于合理的目标域冷启动回退策略。

实现及兼容性差异见 ../docs/reproduction_notes.md。逐轮测试指标保存在 base_training.json、wb_training.json、bb_training.json，不据此选择主报告轮次。
