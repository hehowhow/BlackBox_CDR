# EMCDR comparison, seed 2020

All three runs use the original PFCDR repository's MF-based EMCDR forward,
mapping loss, training loop, and evaluation. Source embeddings are taken from
the real pretrained source MF, not a black-box surrogate. The original random
linear mapping initialization was verified against the stored checkpoint.

Shared settings: seed 2020, original task-specific 80/20 user splits, MF
dimension 10, source/target pretraining 10 epochs, Adam, weight decay 0.
EMCDR mapping: bias-free 10-by-10 linear layer, embedding MSE, Adam lr 0.01,
batch size 128, 10 epochs. Report the fixed final epoch; no test selection.
Existing source/target pretraining checkpoints are reused, matching PFCDR-WB
exactly. This is a mapping-only supplementary run, not a from-raw timing run.

| Task | EMCDR MAE | EMCDR RMSE | PFCDR-WB MAE | BB distill+PFCDR MAE | List encoder MAE |
|---|---:|---:|---:|---:|---:|
| Book -> Movie | 1.176062 | 1.482722 | 0.967869 | 1.049850 | 0.858137 |
| Movie -> Book | 1.119263 | 1.422082 | 1.077062 | 1.279025 | 0.799103 |
| Movie -> Music | 1.653422 | 1.979439 | 1.639227 | 1.649448 | 0.761368 |

Checks passed: source/target MF tensors remained exactly unchanged; cold test
users are excluded from mapping training; test user IDs, item IDs, and labels
match the WB, BB, and list-encoder artifacts row-for-row. The mapping shuffle
RNG is restarted at seed 2020, not resumed from the pretraining RNG state.

Mapping training/evaluation/loading took 12.56, 11.59, and 6.10 seconds,
respectively. These figures exclude reused pretraining and are not end-to-end
training costs. Disk had approximately 1.7 GiB free before and after these runs;
no checkpoint or previous result was deleted.

Per task, full outputs are `results/emcdr_seed2020.json`,
`results/emcdr_training.json`, `results/emcdr_predictions.npz`,
`logs/emcdr_seed2020.log`, and `checkpoints/emcdr_mapping_seed2020.pt`.
Task roots are the project root, `runs/movie_to_book`, and `runs/movie_to_music`.

This is one seed with common fixed settings. The direct list encoder also
differs in architecture, biases and objective; these results do not identify
which component accounts for the gain. In particular, both embedding-transfer
baselines perform poorly on Movie -> Music under the current settings.
