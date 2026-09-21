# List pipeline audit — 2026-09-18

## Findings

No direct target-test-label access was found in List student training, and
independent inference reproduces saved predictions. However, Movie-to-Music
contains cross-domain duplicate user-item-rating records: target test records
are present in source training. Existing performance claims also lack a strong
target-only reference: a training-only target item mean beats List on all three
seed-2020 tasks and on the seed-2021 repeat.

Original experiments, checkpoints and results were not modified. Audit outputs
and fresh diagnostic student runs are isolated under this directory.

## Data and evaluation checks

All four List runs were checked: Book-to-Movie, Movie-to-Book, Movie-to-Music
(seed 2020), and the fresh Movie-to-Music List arm (seed 2021).

- Zero target train/test user overlap; zero mapping-train/test user overlap.
- Target train and test triplets exactly reconstruct their respective parts of
  the pre-split target CSV after independently applying stored raw-ID mappings.
- Overlapping source and target user IDs match their original user identities.
- `train_meta.csv` exactly matches warm-overlap target training triplets.
- Saved evaluation user IDs, item IDs and ratings match `test.csv` row-for-row.
- Predictions and labels are 1D with identical length; no broadcasting issue.
- Independent explicit gather/weighted-sum inference agrees with stored List
  predictions, maximum absolute difference no greater than 1.2e-6.
- For 32 sampled users per run, cached Top-100 scores and ranking thresholds
  agree with the actual source MF checkpoint.
- Book/Movie domains have zero common raw item IDs. Movie/Music have 2,441.

These are scoped checks, not a proof of absence of every possible leakage.
Detailed machine-readable results: `checks.json`.

## Cross-domain contamination

| Movie -> Music split | Test rows | Same raw user/item/rating in source training | Fraction | Affected users |
|---|---:|---:|---:|---:|
| seed 2020 | 93,629 | 5,577 | 5.96% | 1,603 |
| seed 2021 | 90,192 | 5,236 | 5.81% | 1,537 |

The source recommender has already trained on these test interactions under
the other category. Target-user isolation alone does not catch this. All
methods using the same source data are affected; this is not a List-only bug.
Direct rereading of both original gzip archives confirms that all 5,577/5,236
matched records have identical ratings, timestamps, review text and summaries.
They are repeated reviews, not merely different reviews of a shared item.

On just the nonduplicate target test subset, the existing List MAE is 0.762070
(2020) and 0.751203 (2021), close to the full-set values. This does not cleanse
the experiment: the source model was still trained on contaminated data. Clean
results require resolving domain overlap before preprocessing and retraining.
Details: `cross_domain_duplicates.json`; direct raw-review confirmation is
recorded separately in `raw_review_duplicates.json`.

## Training-only target baselines

No validation/test ratings enter these baseline estimates. The item mean is
computed from all target training ratings for that item; unknown items fall
back to the target training global mean. No hyperparameter search was used.

| Task/seed | List MAE | Target item-mean MAE | Global-mean MAE | Training-median MAE |
|---|---:|---:|---:|---:|
| Book -> Movie / 2020 | 0.858137 | 0.833428 | 0.936124 | 0.905495 |
| Movie -> Book / 2020 | 0.799103 | 0.787448 | 0.847054 | 0.819895 |
| Movie -> Music / 2020 | 0.761368 | 0.741995 | 0.829264 | 0.709524 |
| Movie -> Music / 2021 | 0.752058 | 0.746992 | 0.836586 | 0.700783 |

The training median is 5 in all four cases. Movie/Music test ratings are heavily
concentrated at 5. The item mean also beats List in RMSE on all four runs.
Therefore, outperforming the current WB/EMCDR implementations does not yet
demonstrate a gain over a meaningful target-only cold-start baseline.

Using the trained List model with only its global mean and item bias produces
MAE 0.857612, 0.809944, 0.770025, 0.777743 respectively. These are diagnostic
decompositions, not independently trained bias-only models. Permuting cold-user
lists at inference worsens MAE to 0.892579, 0.811931, 0.785194, 0.786012. Thus
the model is sensitive to personal lists, but that does not establish an overall
advantage over target item means. No retrained permutation ablation was run.

## Guarded test-label perturbation replay

Two fresh seed-2021 List students were trained on the same permitted target
training CSV and API cache. One used unchanged test labels; the other shuffled
test labels (52,259 rows changed). These diagnostic runs enable deterministic
PyTorch operations so training nondeterminism cannot mask the comparison.

A Python audit hook denies unexpected project data reads and denies test CSV
reads before the final checkpoint is saved. `torch.load` is prohibited. The
only permitted data inputs are the config, API output, target training CSV and,
after training, the test CSV. Both runs completed with:

- maximum checkpoint parameter difference: **0.0**;
- maximum prediction difference: **0.0**;
- old checkpoint loads: **0**;
- unchanged-label MAE: 0.747957; shuffled-label MAE: 0.875388.

This verifies student training/evaluation separation for the tested path. It
does not validate upstream API data cleanliness; that separately fails the
cross-domain duplication check above. See `replay_check.json` for both replay
directories and their complete training logs. The first diagnostic attempt
stopped because the hook mistook ZIP output creation for an input read; the
guard was corrected and both paired runs were restarted in new directories.

## Cache and timing scope

List training constructs a new model and optimizer; it does not load existing
student weights. Earlier seed-2020 runs intentionally use the already generated
API cache. The separate seed-2021 raw benchmark independently parses raw inputs,
recreates splits, retrains both source models, generates a fresh List API file,
and refuses to use an existing benchmark directory. Recorded code hashes match
the current benchmark scripts. Its WB/List times remain 417.84/115.12 seconds.

OS page cache was not flushed. The full comparison also includes differences
in architecture, batch size (MF 512 versus List 2048), evaluation frequency,
and checkpoint writes. It supports a speed comparison of these implementations,
not a claim of equal-architecture or equally optimized algorithms.

List's vocabulary is built from all queried users, including test-user source
lists. This is a transductive use of unlabeled source-side input under the
existing protocol, not use of target test labels; an inductive claim would
require a separately defined unseen-ID policy. These experiments also do not
enforce a chronological source-before-target split.

## Interpretation and next action

The previous optimistic interpretation should be narrowed. List is fast and
beats the selected WB/EMCDR runs, but currently does not beat simple target-only
item means. Movie/Music also requires cleaning cross-domain duplicate reviews.
No automatic fix or clean retraining was performed during this diagnosis.
First make source/target item spaces disjoint (or otherwise explicitly resolve
duplicate interactions), then retrain all affected methods and retain the
target-only baselines in subsequent comparisons.
