"""Run upstream EMCDR mapping on the same pretrained MF models as PFCDR-WB."""
import argparse
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
os.environ['CDR_RUN_ROOT'] = str(root)

import numpy as np
import pandas as pd
import torch
from experiment import config, seed, load_base, save_json, Run


def main():
    start = time.monotonic()
    cfg = config()
    assert cfg['seed'] == 2020
    out = root / 'results/emcdr_seed2020.json'
    checkpoint = root / 'checkpoints/emcdr_mapping_seed2020.pt'
    assert not out.exists() and not checkpoint.exists(), 'Refuse to overwrite an existing run'
    seed()
    runner = Run(cfg)
    model = load_base(runner, 'wb')
    # The stored full base model contains the original, untrained EMCDR mapping.
    seed()
    initial = runner.get_model()
    torch.testing.assert_close(model.mapping.weight, initial.mapping.weight, rtol=0, atol=0)
    del initial
    seed()
    test_loader, test_ids, train_ids = runner.get_data(runner.co_user_num, inversion=True)
    mapping_loader = runner.read_map_data(train_ids)
    assert not set(train_ids) & set(test_ids)
    optimizer = torch.optim.Adam(model.mapping.parameters(), lr=runner.lr, weight_decay=runner.wd)
    # Upstream EMCDR trains only model.mapping; source/target embeddings stay fixed.
    criterion = torch.nn.MSELoss()
    history = []
    settings = dict(seed=cfg['seed'], task=f'{runner.src}->{runner.tgt}', beta=runner.ratio[1],
                    emb_dim=runner.emb_dim, pretrain_epochs=runner.epoch,
                    mapping_epochs=runner.epoch, mapping_lr=runner.lr,
                    mapping_batch_size=runner.batchsize_map, weight_decay=runner.wd,
                    mapping='upstream Linear(emb_dim, emb_dim, bias=False)',
                    objective='upstream MSE(mapped source user embedding, target user embedding)',
                    source_access='original source embeddings, not surrogate',
                    reused='same seed-2020 pretrained source and target MF as existing PFCDR-WB',
                    reporting='fixed final epoch; no test-based selection')
    print('SETTINGS', json.dumps(settings), flush=True)
    for epoch in range(runner.epoch):
        tick = time.monotonic()
        runner.train(mapping_loader, model, criterion, optimizer, epoch, 'train_map', mapping=True)
        mae, rmse = runner.eval_mae(model, test_loader, 'test_map')
        row = dict(epoch=epoch + 1, mae=mae, rmse=rmse, seconds=time.monotonic() - tick)
        history.append(row)
        save_json(root / 'results/emcdr_training.json', history)
        print('EMCDR_EPOCH', json.dumps(row), flush=True)
    test = pd.read_csv(runner.test_path, header=None).to_numpy()
    x = torch.tensor(test[:, :2], dtype=torch.long, device='cuda')
    model.eval()
    with torch.no_grad():
        predictions = torch.cat([model(batch, 'test_map') for batch in x.split(4096)]).cpu().numpy()
    assert predictions.shape == (len(test),) and np.isfinite(predictions).all()
    references = {}
    for name in ['wb', 'bb', 'list_encoder']:
        data = np.load(root / f'results/{name}_predictions.npz')
        for key, column in [('uid', 0), ('iid', 1), ('y', 2)]:
            np.testing.assert_array_equal(data[key], test[:, column])
        references[name] = float(np.abs(data['prediction'] - data['y']).mean())
    np.savez_compressed(root / 'results/emcdr_predictions.npz', uid=test[:, 0], iid=test[:, 1],
                        y=test[:, 2], prediction=predictions)
    # Verify that the original MF embeddings have not been updated.
    original = torch.load(root / 'checkpoints/base_seed2020.pt', weights_only=True, map_location='cuda')
    for name, tensor in model.state_dict().items():
        if not name.startswith('mapping.'):
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    torch.save(model.mapping.state_dict(), checkpoint)
    error = predictions - test[:, 2]
    result = dict(settings=settings, mae=float(np.abs(error).mean()),
                  rmse=float(np.sqrt(np.square(error).mean())), references_mae=references,
                  test_users=len(np.unique(test[:, 0])), test_ratings=len(test),
                  seconds=time.monotonic() - start,
                  timing_scope='mapping training/evaluation/loading only; reused pretraining excluded',
                  checks=['untrained mapping initialization verified', 'MF weights unchanged',
                          'exact same test samples as all three reference methods'])
    save_json(out, result)
    print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
