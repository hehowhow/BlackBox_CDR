# DNN 源列表在另外两个任务上的复核

每任务三组配对种子；MF/DNN 源模型和 LIST 学生都重新训练；均值 ± 样本标准差，不是预测集成。

| 任务 | 方法 | MAE ↓ | RMSE ↓ |
| --- | --- | ---: | ---: |
| movie_book | item_mean | 0.794258 | 1.034055 |
| movie_book | smoothed_mean_5 | 0.795510 | 1.019479 |
| movie_book | item_median | 0.725972 | 1.136963 |
| movie_book | LIST，源 MF | 0.800825 ± 0.000936 | 1.031850 ± 0.000693 |
| movie_book | LIST，源 DNN | 0.793978 ± 0.000223 | 1.023594 ± 0.002341 |
| movie_music_clean | item_mean | 0.747246 | 1.011406 |
| movie_music_clean | smoothed_mean_5 | 0.760533 | 0.996649 |
| movie_music_clean | item_median | 0.671480 | 1.119439 |
| movie_music_clean | LIST，源 MF | 0.764906 ± 0.001507 | 1.015270 ± 0.002097 |
| movie_music_clean | LIST，源 DNN | 0.751473 ± 0.002812 | 1.008283 ± 0.002324 |

所有 LIST 使用原内积评分，相同初始参数、批次顺序和训练设置，仅源 Top-100 不同。
源/学生均 10 epoch；源固定最后一轮，学生仅以验证 RMSE 选 epoch。全部 12 次训练冻结核验后统一读测试标签。
Movie→Music 沿用清洗数据：源域删除 2441 个共享物品、35333 条评分；再次核对源目标目录交集为零。
本轮不调参、不改变 K、不新增打乱列表对照；数据为历史随机用户划分，无时间过滤。
三个种子同时改变源训练和学生训练，并非两类随机性的全因子分解。
配对用户簇区间条件于已训练的三组模型，没有多重检验校正；不能直接与 Book→Movie 单种子结果视作相同重复量。
细节见 summary.json、per_seed.csv、data_verification.json、各 seed 的 verification.json。
