# Black-box Source-Free Cross-Domain Recommendation

## 1. 研究背景

跨域推荐（Cross-Domain Recommendation, CDR）的基本目标，是利用信息丰富的源域帮助数据稀疏的目标域，从而提升目标域用户的推荐效果。传统 CDR 方法通常需要访问源域用户行为数据、用户表示或者其他中间信息，因此在跨平台场景中存在较强的数据访问要求。

WWW 2025 的 PFCDR（Privacy-Friendly Cross-Domain Recommendation via Distilling User-irrelevant Information）进一步考虑了源域原始数据不可直接共享的问题。PFCDR 不传输源域用户行为日志，而是从已经训练好的推荐模型中通过 conditional model inversion 提取用户对应的 prototype，再利用重叠用户建立 source-target prototype 之间的映射。

因此，PFCDR 实际对应的 source-free 设置可以概括为：

$$
D_s\ \text{不可访问},\qquad
\theta_s\ \text{可访问}.
$$

其中 \(D_s\) 表示源域原始推荐数据，\(\theta_s\) 表示训练好的源域推荐模型。

换言之，PFCDR虽然移除了对源域原始数据的依赖，但仍然默认目标方能够直接访问源推荐模型，包括其参数、embedding 和模型内部计算过程。它需要在源模型内部执行 model inversion，因此本质上属于一种 **white-box source-free CDR**。

这一假设在真实跨平台场景中可能仍然较强。

例如，一个公司可能愿意提供：

$$
API(u)\rightarrow Recommendation
$$

但通常不会直接向另一个公司交付：

$$
\theta_s,\quad
UserEmbedding,\quad
ItemEmbedding,\quad
Gradient.
$$

因此，一个更加受限但现实的场景是：

$$
D_s\ \text{不可访问},
$$

$$
\theta_s\ \text{不可访问},
$$

只能够调用源推荐系统并获得推荐结果。

我们将这一问题暂时定义为：

# Black-box Source-Free Cross-Domain Recommendation

---

## 2. Black-box Source-Free CDR 的基本设定

为了避免同时引入过多困难，当前阶段仍然保留 PFCDR 的 overlapping-user 假设：

$$
U_s\cap U_t\neq\emptyset.
$$

也就是说，目前只改变一个条件：

### White-box Source-Free CDR

可以访问：

$$
f_s,\theta_s
$$

不能访问：

$$
D_s.
$$

### Black-box Source-Free CDR

不能访问：

$$
D_s,\theta_s,
$$

只能查询：

$$
f_s(u)\rightarrow
Top\text{-}K\ recommendations.
$$

如果实验实现需要，也可以允许 API 返回预测分数，但不能访问：

* source model parameters；
* user embeddings；
* item embeddings；
* gradients；
* hidden representations；
* source raw interactions。

因此，两种 setting 的唯一区别应该尽量控制为：

$$
\boxed{\text{是否可以访问 source model 内部信息}}
$$

而不是同时改变数据集、推荐模型或其他实验条件。

---

## 3. 为什么这个问题有研究依据

Black-box adaptation 并不是一个人为构造出来的问题。

例如 CVPR 2022 的 DINE 已经研究了 black-box domain adaptation：源数据不可访问，源模型参数也不可访问，只能获得 source predictor 的预测结果，然后利用这些输出完成目标域适配。

在推荐系统领域，NeurIPS 2025 的 DBGRME 又进一步证明：

> 即使没有 victim recommender 的训练数据，仅通过查询 black-box recommender，也可以提取其推荐知识并训练 surrogate recommender。

DBGRME 专门研究 data-free black-box recommender model extraction。

因此，目前已经分别存在：

$$
\text{Black-box Domain Adaptation}
$$

和

$$
\text{Data-free Black-box Recommendation Model Extraction}.
$$

PFCDR 则证明：

$$
\text{Source-free Cross-Domain Recommendation}
$$

是一个有效的推荐研究问题。

但在本轮检索中，暂未发现一个直接研究：

$$
\boxed{
\text{source data unavailable}
+
\text{source model black-box}
+
\text{cross-domain recommendation}
}
$$

并以**提高目标域推荐效果**为最终任务的同构工作。

因此首先需要通过实验确认：

> 从 white-box source-free 转向 black-box source-free 后，是否真的会出现明显的推荐性能损失。

如果没有明显损失，那么单独定义 Black-box Source-Free CDR 的必要性会比较弱。

如果存在明显损失，则说明 PFCDR 尚未解决模型不可共享条件下的 source-free CDR，这才构成后续设计新方法的基础。

---

# 4. Baby Check 的唯一核心问题

当前实验暂时不回答：

* black-box knowledge 为什么有效；
* 哪种信息最值得查询；
* 如何选择用户；
* 如何设计 query strategy；
* query budget 如何分配；
* source-target similarity 如何衡量；
* 应该迁移什么 representation。

这些都不是当前 baby check 的目标。

当前只回答：

$$
\boxed{
\text{White-box access}
\rightarrow
\text{Black-box access}
}
$$

之后：

$$
\boxed{
\text{CDR recommendation performance 是否出现明显下降？}
}
$$

因此整个 baby check 可以压缩为三个核心实验组。

---

# 5. 三个核心实验组

## Group 1：Target-only

只利用目标域数据：

$$
D_t
$$

训练目标推荐模型，不进行任何跨域知识迁移。

记为：

$$
\text{TGT}.
$$

它代表：

> 完全不使用 source knowledge 时的推荐效果。

PFCDR 官方代码已经提供 TGTOnly baseline。

---

## Group 2：White-box PFCDR

直接按照 PFCDR 官方实现运行。

可以访问训练完成的 source recommender：

$$
f_s,\theta_s.
$$

利用 source model inversion 获得 prototypes，并进行后续跨域迁移。

记为：

$$
\text{PFCDR-WB}.
$$

它代表：

> source data 不可访问，但 source model 完全开放时能够达到的性能。

这是当前实验中的 **white-box reference**。

---

## Group 3：Black-box PFCDR

这一组模拟我们真正关心的情况。

首先训练与 Group 2 完全相同的 source recommender。

然后把它封装为 black-box：

$$
f_s(u)\rightarrow Recommendation.
$$

后续程序不能再读取 source model 的：

$$
\theta_s,\quad embeddings,\quad gradients.
$$

由于 PFCDR 本身无法直接在 black-box source model 上执行 model inversion，因此当前 baby check **不需要提出新的 Black-box CDR 方法**。

采用最直接的已有技术路线即可：

$$
\text{Black-box Source Model}
$$

$$
\downarrow
$$

$$
\text{Existing Black-box Model Extraction}
$$

$$
\downarrow
$$

$$
\text{Surrogate Source Model}
$$

$$
\downarrow
$$

$$
\text{PFCDR}
$$

记为：

$$
\text{PFCDR-BB}.
$$

优先使用现有的 DBGRME 等 data-free black-box recommender extraction 方法获得 surrogate model。DBGRME 的目标正是在无法访问训练数据的情况下，根据 black-box recommender 的查询结果恢复其推荐行为。

需要强调：

**PFCDR-BB 并不是我们提出的新方法。**

它只是一个用于 baby check 的朴素解决方案：

> 当 source model 不可访问时，先尽可能恢复一个 surrogate source model，再按照现有 PFCDR 进行跨域迁移。

当前阶段只使用它来估计：

$$
\text{Black-box access造成多大的性能损失。}
$$

---

# 6. 最小实验设置

为了尽可能减少工作量，所有设置直接沿用 PFCDR。

PFCDR 使用 Amazon 5-core 数据的三个跨域任务：

$$
Book\rightarrow Movie
$$

$$
Book\rightarrow Music
$$

$$
Music\rightarrow Movie.
$$

三个任务分别包含 37,388、16,738 和 18,031 个 overlapping users。

PFCDR 官方代码同时支持 MF、GMF 和 YouTube DNN 三种基础推荐模型。

### 第一轮 Baby Check

先只使用：

$$
\boxed{Book\rightarrow Movie}
$$

和：

$$
\boxed{MF}
$$

进行实验。

不要一开始跑全部模型。

数据划分、cold-start user 设置以及评价指标全部沿用 PFCDR 官方设置，不改变实验协议。

PFCDR 原实验使用 MAE 评价推荐误差，并分别测试不同 cold-start 用户比例；其通用性实验也固定在 \(\beta=20\%\) 下比较 MF、GMF 和 YouTube DNN。

因此第一轮直接固定：

$$
\boxed{\beta=20\%}
$$

即可。

这样第一轮实际上只需要得到三个数：

| Setting  | Source access          | Result |
| -------- | ---------------------- | -----: |
| TGT      | 无 source information   |    MAE |
| PFCDR-WB | White-box source model |    MAE |
| PFCDR-BB | Black-box source model |    MAE |

其中：

$$
MAE\downarrow
$$

代表推荐性能更好。

---

# 7. 第一轮实验最想看到什么结果

最理想的结果是：

$$
\boxed{
MAE_{PFCDR-WB}
<
MAE_{PFCDR-BB}
<
MAE_{TGT}
}
$$

即：

### White-box PFCDR 最好

说明 source knowledge 确实能够帮助 target recommendation。

### Black-box PFCDR 比 White-box PFCDR 差

说明：

$$
\text{model accessibility}
$$

确实造成了明显的信息损失。

这就是我们要寻找的 **research gap**。

### 但 Black-box PFCDR 又优于 Target-only

说明：

> black-box source model 中仍然存在可利用的跨域知识。

因此问题虽然困难，但并不是完全不可解。

这实际上是 baby check 最漂亮的结果。

---

# 8. 建议额外报告两个非常简单的数字

除了原始 MAE，可以计算：

## White-box Transfer Gain

$$
G_{WB}
=
MAE_{TGT}
-
MAE_{PFCDR-WB}
$$

表示 white-box source knowledge 带来的收益。

## Black-box Transfer Gain

$$
G_{BB}
=
MAE_{TGT}
-
MAE_{PFCDR-BB}.
$$

然后计算：

$$
RecoveryRatio
=
\frac{G_{BB}}{G_{WB}}.
$$

例如：

如果：

$$
MAE_{TGT}=1.20
$$

$$
MAE_{WB}=1.00
$$

$$
MAE_{BB}=1.15,
$$

则：

$$
G_{WB}=0.20,
$$

$$
G_{BB}=0.05,
$$

因此：

$$
RecoveryRatio=25\%.
$$

直观含义就是：

> 现有黑盒方案只能恢复 white-box CDR 大约 25% 的跨域收益。

这种数字比单独说“性能下降 0.15”更容易判断问题空间。

---

# 9. 如何根据结果决定是否继续

## 情况 A：最理想

$$
PFCDR\text{-}WB
>
PFCDR\text{-}BB
>
TGT.
$$

这里的 “>” 表示推荐效果更好。

说明：

1. source knowledge 有价值；
2. black-box source knowledge仍然有价值；
3. 但现有 black-box 方式无法充分恢复 white-box transfer ability。

这是最值得继续的结果。

可以直接判：

$$
\boxed{\text{GO}}
$$

后续真正的论文问题就可以变成：

> 如何直接从 black-box recommender 中获取对 target domain 有价值的知识，而不是先恢复整个 source recommender？

---

## 情况 B：Black-box 与 White-box 几乎一样

$$
PFCDR\text{-}BB
\approx
PFCDR\text{-}WB
<
TGT.
$$

这反而不太理想。

说明现有 model extraction 已经基本消除了 black-box restriction。

那么：

> “source model 只能黑盒访问”

本身可能不足以形成很强的新研究问题。

需要谨慎继续。

---

## 情况 C：Black-box 与 Target-only 接近

$$
PFCDR\text{-}WB
<
PFCDR\text{-}BB
\approx
TGT.
$$

说明：

white-box source knowledge 很有效，

但是现有简单的：

$$
ModelExtraction+PFCDR
$$

在 black-box 环境下几乎无法发挥作用。

这其实仍然是一个有意思的结果，但只能说明：

$$
\boxed{\text{problem很难}}
$$

还不能说明：

$$
\boxed{\text{problem可解}}
$$

此时可以再换一个更强的 black-box extraction baseline 验证一次。

如果仍然如此，再决定是否投入新 method。

---

## 情况 D：PFCDR 本身对 Target-only 提升很小

如果：

$$
PFCDR\text{-}WB
\approx TGT,
$$

那么这个具体 dataset / setting 就不适合用来证明 Black-box Source-Free CDR。

因为连 white-box source transfer 本身都没有多少价值，就没有必要讨论 black-box 后损失了多少。

---

# 10. 第一轮之后再决定是否扩大实验

第一轮暂时只跑：

$$
Book\rightarrow Movie
$$

$$
MF
$$

$$
\beta=20\%.
$$

如果能够观察到明显的：

$$
PFCDR\text{-}WB
>
PFCDR\text{-}BB
>
TGT
$$

趋势，再扩展到：

$$
Book\rightarrow Music
$$

和：

$$
Music\rightarrow Movie.
$$

如果三个任务趋势基本一致，baby check 就已经足够。

此时暂时仍然**没有必要**做：

* Query budget；
* Top-K sensitivity；
* source-target similarity；
* user-level分析；
* 不同 API 返回形式；
* 不同 surrogate architecture；
* mechanism ablation；
* transferable knowledge visualization。

这些实验应该等真正提出 Black-box Source-Free CDR 方法之后，再围绕具体 method 设计。

---

# 最终需要外部 Agent 回答的问题

最终不要提交大量分析。

只需要回答以下三个问题：

### Q1

White-box PFCDR 相比 Target-only 是否能够获得明显的跨域推荐收益？

### Q2

把 source model 改成只能黑盒访问以后，现有简单方案的推荐性能是否明显下降？

### Q3

黑盒情况下是否仍然保留一定的跨域推荐收益？

最希望得到的结论是：

$$
\boxed{
WhiteBox
>
BlackBox
>
TargetOnly
}
$$

如果这个关系在多个 domain pair 上稳定存在，就可以初步说明：

> **PFCDR 已经证明 source-free CDR 可行，但其效果依赖于较强的 white-box source-model access。当 source model 只能作为服务接口被调用时，现有方案存在明显性能缺口；与此同时，black-box model 中仍然保留可用于跨域推荐的知识。因此 Black-box Source-Free CDR 具有进一步研究空间。**
