# List encoder: minimal potential check

> Historical initial result. Later auditing found that target-only item means
> outperform List. See [current analysis](experiment_progress.md) before
> interpreting the original potential-check conclusion.

Goal: compare one directly trained black-box list encoder against existing
PFCDR-WB (MAE 0.967869) and score-distillation + PFCDR (1.049850).
No distillation loss, inversion, ablations, validation sweep, or retraining of
references. Existing seed-2020 Book-to-Movie split and Top-100 API cache remain
unchanged. Report the fixed final epoch, not the best test epoch.

Model: learned 32-dimensional embeddings of returned opaque source item IDs;
normalized reciprocal-log rank pooling; MLP 32 -> 64 -> 32 with ReLU and dropout
0.1; target item dot product plus global training mean and learned item bias.
Source scores are not used. No independently trainable user-ID embeddings.
All source item embeddings are learned locally from random initialization.

Training: all 1,541,404 target training ratings, Adam lr=0.001, weight decay
0.00001, batch size 2048, 10 epochs, seed 2020. Users without a cached source
list have zero user representation, so their observations train target item
biases only. These users do not obtain fabricated source lists. Cold test users
have genuine cached API lists; their target ratings are opened only after the
final checkpoint is saved. Predictions are not clipped.

The learner reads no source interactions or teacher checkpoints. The cached
lists are registered fixed buffers, with compact source IDs and a padding ID.
This script uses explicit input paths, not the earlier extraction sandbox.

Run: `/root/miniconda3/bin/python -u scripts/run_list_encoder.py`.
Outputs: `results/list_encoder_summary.json`, `results/list_encoder_training.json`,
`results/list_encoder_predictions.npz`, `checkpoints/list_encoder.pt`.

Interpretation: a quick feasibility comparison, not an architecture-controlled
or multi-seed result. Differences also include biases, embedding dimension and
direct target supervised training. A favorable MAE alone does not isolate the
benefit of personalized source information.

## Completed result

Fixed epoch 10 MAE: **0.858137**; RMSE: **1.123731**. The same 7,478
cold users / 156,129 test ratings were checked row-for-row against the WB
prediction artifact. MAE is 0.109732 below WB and 0.191713 below the previous
black-box baseline. Training plus evaluation took 18.98 seconds (excluding
interpreter startup). This supports further investigation, not a claim of
mechanistic attribution or robust superiority across seeds/tasks.
