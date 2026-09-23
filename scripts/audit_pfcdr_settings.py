"""Read-only comparison of PFCDR defaults, stored runs and paper transcriptions.

No training, test-label loading, or checkpoint selection is performed. Historical
best-test values are descriptive diagnostics, never newly selected results.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parents[1]
UPSTREAM = BASE / 'external/PFCDR'
OUT = BASE / 'results/paper_config_audit_20260922'
PAPER = Path('/home/hehao32/3696410.3714580.pdf')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    upstream = read(UPSTREAM / 'config.json')
    inversion = read(UPSTREAM / 'inversion_config.json')
    defaults = {}
    for node in ast.walk(ast.parse((UPSTREAM / 'main.py').read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            for kw in node.keywords:
                if kw.arg == 'default':
                    defaults[ast.literal_eval(node.args[0])] = ast.literal_eval(kw.value)
    model_tree = ast.parse((UPSTREAM / 'models.py').read_text())
    mf = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == 'MFBasedModel')
    constants = [ast.literal_eval(n.value) for n in ast.walk(mf) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'topk' for t in n.targets)]
    assert constants == [70000]
    k = constants[0]
    compared_files = [BASE / 'configs/first_round.json', BASE / 'scripts/experiment.py',
                      BASE / 'scripts/run_emcdr.py', BASE / 'scripts/run_list_encoder.py',
                      BASE / 'scripts/residual_experiment.py', BASE / 'scripts/representation_diagnostic.py',
                      UPSTREAM / 'config.json', UPSTREAM / 'main.py', UPSTREAM / 'models.py',
                      UPSTREAM / 'run.py', UPSTREAM / 'inversion.py', UPSTREAM / 'inversion_config.json',
                      UPSTREAM / 'preprocessing.py']
    runs = []
    for index, rel in enumerate(['.', 'runs/movie_to_book', 'runs/movie_to_music']):
        root = BASE / rel
        cfg = dict(upstream)
        cfg.update(read(root / 'configs/first_round.json'))
        pair = cfg['src_tgt_pairs'][cfg['task']]
        expected = dict(base_model=defaults['--base_model'], epoch=defaults['--epoch'],
                        lr=defaults['--lr'], lr_prototype=defaults['--lr_prototype'],
                        emb_dim=upstream['emb_dim'], wd=upstream['wd'])
        for key in ['iter', 'batch', 'lr', 'weight_decay', 'target_label']:
            expected['inversion_' + key] = inversion[key]
        matches = {key: dict(local=cfg[key], public_default=value, equal=cfg[key] == value)
                   for key, value in expected.items()}
        histories = {}
        for name in ['base', 'wb', 'emcdr']:
            path = root / f'results/{name}_training.json'
            rows = read(path)
            compared_files.append(path)
            if name == 'base':
                histories[name] = {domain: [r['epoch'] for r in rows if r['domain'] == domain]
                                   for domain in ['src', 'tgt']}
            else:
                histories[name] = dict(epochs=[r['epoch'] for r in rows], final=rows[-1],
                                       descriptive_min_test_mae=min(rows, key=lambda r: r['mae']),
                                       descriptive_min_test_rmse=min(rows, key=lambda r: r['rmse']))
        audit_path = BASE / f'results/mean_leakage_audit/case_{index}.json'
        audit = read(audit_path)
        compared_files += [root / 'configs/first_round.json', audit_path]
        means = audit['historical_item_mean_metrics']
        n = pair['uid_tgt']
        runs.append(dict(root=rel, task=pair['src']+'->'+pair['tgt'], configured=cfg,
                         common_default_comparison=matches, history=histories,
                         audited_item_mean=means, target_users=n, effective_voting_k=k,
                         voting_fraction=k/n, minimum_pairwise_voter_intersection=max(0, 2*k-n),
                         minimum_pairwise_voter_overlap_fraction=max(0, 2*k-n)/k,
                         official_task_direction_match=any(p['src'] == pair['src'] and p['tgt'] == pair['tgt']
                                                           for p in upstream['src_tgt_pairs'].values()),
                         artifact_availability={p: (root / p).exists() for p in
                                                ['checkpoints/base_seed2020.pt', 'results/wb_predictions.npz']}))
    recent = BASE / 'runs/residual_study_20260922'
    diagnostic = BASE / 'runs/representation_diagnostic_8571'
    compared_files += [recent / 'experiment.json', diagnostic / 'protocol.json']
    result = dict(
        scope='Static code/configuration audit and arithmetic over stored metrics; no causal retraining',
        paper=dict(path=str(PAPER), sha256=sha(PAPER),
                   transcription='Manually read local PDF; tables/figures visually checked on PDF pages 6, 7, 8, 11',
                   tasks=['Books->Movies_and_TV', 'Books->CDs_and_Vinyl', 'CDs_and_Vinyl->Movies_and_TV'],
                   pretrain_lr_grid=[.001, .005, .01, .02, .1], embedding_dim=10, runs=10,
                   test_overlap_fractions=[.2, .5, .8],
                   table1_book_movie_beta20=dict(TGT=dict(mae=4.1831, rmse=4.7536),
                       EMCDR=dict(mae=1.1651, rmse=1.4548), PTUPCDR=dict(mae=.9970, rmse=1.3317),
                       PFCDR=dict(mae=.9579, rmse=1.2437)),
                   figure2_book_movie_pfcdR_mae=dict(MF=.958, GMF=.867, DNN=.802),
                   figure3_k_values=[100, 1000, 10000, 30000, 50000, 70000],
                   paper_inversion_loss_text='BCE (Section 5.1), but squared error in Eqs. 9/10 and Appendix A',
                   exact_best_lr_per_task_and_backbone='not specified in inspected paper/config files'),
        upstream=dict(commit=subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
                      argparse_defaults=defaults, config=upstream, inversion=inversion,
                      effective_mf_voting_k=k, report_policy='separate minima of test MAE and test RMSE across epochs'),
        historical_runs=runs, recent_protocol=read(recent / 'experiment.json'),
        diagnostic_protocol=read(diagnostic / 'protocol.json'),
        recent_manifests={task: {key: value for key, value in read(recent / task / 'manifest.json').items()
                                if key not in ['hashes', 'baseline_hashes']}
                          for task in ['book_movie', 'movie_book', 'movie_music_clean']},
        evidence_sha256={str(p.relative_to(BASE)): sha(p) for p in compared_files},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    for r in runs:
        assert all(v['equal'] for v in r['common_default_comparison'].values())
        assert all(e == list(range(1, 11)) for e in r['history']['base'].values())
        print(r['task'], 'public common defaults match;', 'vote fraction', round(r['voting_fraction'], 4),
              'PFCDR final / best-test MAE', r['history']['wb']['final']['mae'],
              r['history']['wb']['descriptive_min_test_mae']['mae'])
    print('AUDIT_COMPLETE', OUT / 'audit.json')


if __name__ == '__main__':
    main()
