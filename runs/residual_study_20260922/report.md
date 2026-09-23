# 三组跨域残差对照实验

状态：PASS。所有训练完成、验证选择冻结后，统一读取测试标签评价。

每组任务一个固定 60/20/20 用户划分、三个训练种子；结果为三次单模型指标的均值及样本标准差，不是预测集成。
主指标为 RMSE；MAE 同时报告。残差模型以验证 RMSE 选择 epoch，包含零修正的 epoch 0。

| 任务 | 固定基线 RMSE | 目标偏置 RMSE | 真实源 RMSE | 打乱源 RMSE | 真实源 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| book_movie | 1.086142 | 1.086523 ± 0.000067 | 1.085331 ± 0.000306 | 1.085950 ± 0.000576 | 0.836407 ± 0.000393 |
| movie_book | 1.019479 | 1.019479 ± 0.000000 | 1.019479 ± 0.000000 | 1.019479 ± 0.000000 | 0.795510 ± 0.000000 |
| movie_music_clean | 0.996649 | 0.996649 ± 0.000000 | 0.996649 ± 0.000000 | 0.996649 ± 0.000000 | 0.760533 ± 0.000000 |

## book_movie

清洗记录：`{"rule": "No filtering: source and target raw item catalogs already disjoint", "removed_items": 0, "removed_source_ratings": 0, "preserved_prior_protocol": true, "prior_test_sha256": "57713ab0a4eeb6461b73f776008d6e8a4ca2694bd0946dcc747600d32fce9665"}`

验证集选定基线：`smoothed_mean_5`。选定 epoch：`{"target_bias": {"2020": 15, "2021": 12, "2022": 19}, "real": {"2020": 2, "2021": 3, "2022": 2}, "shuffled": {"2020": 3, "2021": 2, "2022": 2}}`。

| 差异（负数有利于真实源） | ΔRMSE | 用户簇 bootstrap 95% CI |
| --- | ---: | --- |
| real_minus_baseline | -0.000811 | [-0.001701, 0.000077] |
| real_minus_shuffled | -0.000618 | [-0.001250, -0.000098] |
| real_minus_target_bias | -0.001192 | [-0.001859, -0.000528] |

本轮主指标上的源域增量证据：未同时满足优于固定基线、打乱源和目标偏置的条件。

## movie_book

清洗记录：`{"rule": "Drop all shared raw item IDs from source before user splitting; target untouched; no reapplication of 5-core", "removed_items": 0, "removed_source_ratings": 0, "source_users_before": 123960, "source_users_after": 123960, "overlap_users_before": 37388, "remaining_cross_domain_raw_items": 0}`

验证集选定基线：`smoothed_mean_5`。选定 epoch：`{"target_bias": {"2020": 0, "2021": 0, "2022": 0}, "real": {"2020": 0, "2021": 0, "2022": 0}, "shuffled": {"2020": 0, "2021": 0, "2022": 0}}`。

| 差异（负数有利于真实源） | ΔRMSE | 用户簇 bootstrap 95% CI |
| --- | ---: | --- |
| real_minus_baseline | 0.000000 | [0.000000, 0.000000] |
| real_minus_shuffled | 0.000000 | [0.000000, 0.000000] |
| real_minus_target_bias | 0.000000 | [0.000000, 0.000000] |

本轮主指标上的源域增量证据：未同时满足优于固定基线、打乱源和目标偏置的条件。

## movie_music_clean

清洗记录：`{"rule": "Drop all shared raw item IDs from source before user splitting; target untouched; no reapplication of 5-core", "removed_items": 2441, "removed_source_ratings": 35333, "source_users_before": 123960, "source_users_after": 123803, "overlap_users_before": 18031, "remaining_cross_domain_raw_items": 0}`

验证集选定基线：`smoothed_mean_5`。选定 epoch：`{"target_bias": {"2020": 0, "2021": 0, "2022": 0}, "real": {"2020": 0, "2021": 0, "2022": 0}, "shuffled": {"2020": 0, "2021": 0, "2022": 0}}`。

| 差异（负数有利于真实源） | ΔRMSE | 用户簇 bootstrap 95% CI |
| --- | ---: | --- |
| real_minus_baseline | 0.000000 | [0.000000, 0.000000] |
| real_minus_shuffled | 0.000000 | [0.000000, 0.000000] |
| real_minus_target_bias | 0.000000 | [0.000000, 0.000000] |

本轮主指标上的源域增量证据：未同时满足优于固定基线、打乱源和目标偏置的条件。

## 范围与复核

Book→Movie 复用 8564 划分；其他任务采用稳定原始 ID 排序后固定用户划分。
Movie→Music 在划分之前删除源电影域所有与音乐域共享 ASIN 的记录，保留音乐目标域；处理后物品交集和跨域同用户物品评分重叠均为零。
清洗后没有重新做 5-core。该规则改变源域目录、历史量和重叠用户集合，清洗前后指标不能直接用于归因。
全部任务仍为随机用户切分，没有时间过滤；完整目录 ID 编码和源查询包含冷用户，不是严格未知目录的归纳协议。
学生只接收源 MF Top-100 ID 列表；每种子重新训练源 MF，共享给配对真实/打乱学生。训练进程禁止读取测试评分和源模型权重。
CI 是固定三个训练种子下的测试用户抽样不确定性；种子稳定性另见标准差。多任务比较未做多重检验校正，不作普遍性结论。
中位数是 MAE 参照，不能根据测试结果改选目标或 epoch。细分指标和全部种子结果见各任务 results.json。
