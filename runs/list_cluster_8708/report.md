# LIST 共现聚类聚合实验

Music→Movie去重版；β为测试冷用户比例；三种种子；均值±样本标准差，越低越好。
训练用户Top100构建物品×用户的排名加权稀疏矩阵，物品行L2归一化，SVD32降维后再次行归一化，KMeans聚为三类。选择阶段只拟合inner_train，最终重训重新拟合outer_train。未知源ID不参与聚合，空簇为零。
cooccurrence按物品簇分组；size_matched_random在每个用户内打乱已知物品的簇标签，严格保持每用户三组大小及未知位置。两个新模型具有相同初始化、参数量和训练流程；与排名分段同为96→64→32，比全局池化多4096参数。
仅使用源API列表ID及排名，无真实源评分、教师权重或源内容；训练文件访问保护禁止测试标签。24个模型全部冻结后统一评价。未调簇数、降维或学习率；沿用相同验证选epoch和lambda、全量重训、错配用户列表及三个种子。
历史测试已使用，本轮为探索。固定随机对照保留了聚类产生的组大小信息；它检验组内物品归属的作用，不是完全去除聚类信息的对照。

| β | 方法 | MAE ± 标准差 | RMSE ± 标准差 |
| --- | --- | ---: | ---: |
| 20% | item_mean | 0.828612 ± 0.000000 | 1.074618 ± 0.000000 |
| 20% | EMCDR_MF | 1.315861 ± 0.053348 | 1.719547 ± 0.077318 |
| 20% | PFCDR_MF | 0.992507 ± 0.021103 | 1.262861 ± 0.024545 |
| 20% | EMCDR_DNN | 0.819904 ± 0.008216 | 1.069249 ± 0.007448 |
| 20% | PFCDR_DNN | 0.817090 ± 0.019706 | 1.073654 ± 0.003413 |
| 20% | LIST_original | 0.822822 ± 0.008570 | 1.067575 ± 0.001593 |
| 20% | LIST_residual_real | 0.808497 ± 0.005254 | 1.056126 ± 0.001299 |
| 20% | LIST_residual_shuffled | 0.826968 ± 0.001146 | 1.073404 ± 0.000246 |
| 20% | rank_segments_real | 0.810930 ± 0.003371 | 1.058031 ± 0.002235 |
| 20% | rank_segments_shuffled | 0.827506 ± 0.000100 | 1.073792 ± 0.000315 |
| 20% | random_groups_real | 0.811006 ± 0.002473 | 1.059973 ± 0.000668 |
| 20% | random_groups_shuffled | 0.827933 ± 0.001000 | 1.074374 ± 0.000988 |
| 20% | cooccurrence_real | 0.809601 ± 0.005720 | 1.057300 ± 0.002526 |
| 20% | cooccurrence_shuffled | 0.827196 ± 0.000381 | 1.073555 ± 0.000343 |
| 20% | size_matched_random_real | 0.813166 ± 0.001746 | 1.060641 ± 0.001004 |
| 20% | size_matched_random_shuffled | 0.827302 ± 0.000204 | 1.073932 ± 0.000139 |
| 80% | item_mean | 0.840744 ± 0.000000 | 1.097641 ± 0.000000 |
| 80% | EMCDR_MF | 1.352129 ± 0.011181 | 1.779450 ± 0.023028 |
| 80% | PFCDR_MF | 1.048391 ± 0.010797 | 1.391033 ± 0.011732 |
| 80% | EMCDR_DNN | 0.826689 ± 0.007726 | 1.097017 ± 0.004297 |
| 80% | PFCDR_DNN | 0.849226 ± 0.033258 | 1.140535 ± 0.028757 |
| 80% | LIST_original | 0.870605 ± 0.002497 | 1.124753 ± 0.008310 |
| 80% | LIST_residual_real | 0.833717 ± 0.001640 | 1.089396 ± 0.001209 |
| 80% | LIST_residual_shuffled | 0.840470 ± 0.000476 | 1.097543 ± 0.000157 |
| 80% | rank_segments_real | 0.832636 ± 0.000907 | 1.089779 ± 0.000437 |
| 80% | rank_segments_shuffled | 0.840389 ± 0.000244 | 1.097504 ± 0.000076 |
| 80% | random_groups_real | 0.834095 ± 0.001768 | 1.091244 ± 0.001661 |
| 80% | random_groups_shuffled | 0.840471 ± 0.000314 | 1.097539 ± 0.000106 |
| 80% | cooccurrence_real | 0.834123 ± 0.000967 | 1.090821 ± 0.000592 |
| 80% | cooccurrence_shuffled | 0.840363 ± 0.000276 | 1.097505 ± 0.000082 |
| 80% | size_matched_random_real | 0.833938 ± 0.000538 | 1.091286 ± 0.000921 |
| 80% | size_matched_random_shuffled | 0.840430 ± 0.000381 | 1.097529 ± 0.000118 |

## 配对差异（新方法减参照）

95%区间按测试用户成组重采样2000次，先计算各种子指标再求均值；条件于已训练模型与当前划分，未做多重比较校正。负值代表新方法更好。

| β | 新方法 | 参照 | ΔRMSE | 95%区间 |
| --- | --- | --- | ---: | --- |
| 20% | cooccurrence_real | item_mean | -0.017318 | [-0.022423, -0.012715] |
| 20% | cooccurrence_real | LIST_original | -0.010275 | [-0.015896, -0.003851] |
| 20% | cooccurrence_real | rank_residual | +0.001174 | [-0.000104, +0.002425] |
| 20% | cooccurrence_real | size_matched_random_real | -0.003342 | [-0.004984, -0.001867] |
| 20% | cooccurrence_real | rank_segments_real | -0.000731 | [-0.002136, +0.000524] |
| 20% | cooccurrence_real | shuffled | -0.016255 | [-0.021544, -0.011454] |
| 20% | size_matched_random_real | item_mean | -0.013977 | [-0.018314, -0.010170] |
| 20% | size_matched_random_real | LIST_original | -0.006934 | [-0.013118, +0.000320] |
| 20% | size_matched_random_real | rank_residual | +0.004515 | [+0.002860, +0.006310] |
| 20% | size_matched_random_real | size_matched_random_real | +0.000000 | [+0.000000, +0.000000] |
| 20% | size_matched_random_real | rank_segments_real | +0.002610 | [+0.001182, +0.004074] |
| 20% | size_matched_random_real | shuffled | -0.013291 | [-0.017917, -0.009290] |
| 80% | cooccurrence_real | item_mean | -0.006820 | [-0.007969, -0.005701] |
| 80% | cooccurrence_real | LIST_original | -0.033931 | [-0.039047, -0.029298] |
| 80% | cooccurrence_real | rank_residual | +0.001426 | [+0.001062, +0.001817] |
| 80% | cooccurrence_real | size_matched_random_real | -0.000465 | [-0.001011, +0.000139] |
| 80% | cooccurrence_real | rank_segments_real | +0.001042 | [+0.000512, +0.001554] |
| 80% | cooccurrence_real | shuffled | -0.006683 | [-0.007829, -0.005574] |
| 80% | size_matched_random_real | item_mean | -0.006355 | [-0.007452, -0.005297] |
| 80% | size_matched_random_real | LIST_original | -0.033466 | [-0.038614, -0.028756] |
| 80% | size_matched_random_real | rank_residual | +0.001891 | [+0.001357, +0.002407] |
| 80% | size_matched_random_real | size_matched_random_real | +0.000000 | [+0.000000, +0.000000] |
| 80% | size_matched_random_real | rank_segments_real | +0.001507 | [+0.000912, +0.002099] |
| 80% | size_matched_random_real | shuffled | -0.006243 | [-0.007341, -0.005193] |