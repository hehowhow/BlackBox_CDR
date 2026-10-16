"""Export the verified 54-cell MF comparison to CSV, LaTeX and its compiled PDF."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

METHODS = ['TGTOnly','EMCDR','PFCDR','LIST_V1','LIST_V1_Langevin','LAPT']
TASKS = ['book_movie','book_music','music_movie_clean']
LABELS = {'book_movie':'Book $\\rightarrow$ Movie','book_music':'Book $\\rightarrow$ Music',
          'music_movie_clean':'Music $\\rightarrow$ Movie'}
TEXT_LABELS = {'book_movie':'Book→Movie','book_music':'Book→Music','music_movie_clean':'Music→Movie（去重）'}
HEADERS = ['TGTOnly','EMCDR','PFCDR','LIST V1','LIST V1 + Langevin','LAPT']


def export(root):
    assert json.loads((root/'verification.json').read_text())['status']=='PASS'
    with (root/'results.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==54
    index={(r['task'],int(r['beta']),r['method']):r for r in rows}
    tables=[]
    md=['# MF 全量实验：seed 2020','',
        '三个任务，β=20%/50%/80%，六种方法，共54组方法结果。数值越小越好。',
        '所有方法在同一任务内使用同一数据划分；三个LIST方法共享新训练的MF源Top-100。','']
    for metric in ['mae','rmse']:
        body=[r'\begin{table}[htbp]',r'\centering',
              '\\caption{'+metric.upper()+r' on three cross-domain tasks (MF, seed 2020). Lower is better.}',
              r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{llrrrrrr}',r'\toprule',
              'Task & $\\beta$ & '+' & '.join(HEADERS)+r' \\',r'\midrule']
        md += ['## '+metric.upper(),'',
               '| 数据集 | β | '+' | '.join(HEADERS)+' |',
               '| --- | ---: | '+' | '.join(['---:']*6)+' |']
        for ti,task in enumerate(TASKS):
            for beta in [20,50,80]:
                values=[float(index[task,beta,m][metric]) for m in METHODS]
                best=min(values)
                tex=[f'\\textbf{{{v:.4f}}}' if abs(v-best)<1e-12 else f'{v:.4f}' for v in values]
                body.append((LABELS[task] if beta==20 else '')+f' & {beta}\\% & '+' & '.join(tex)+r' \\')
                vals=[f'**{v:.6f}**' if abs(v-best)<1e-12 else f'{v:.6f}' for v in values]
                md.append('| '+TEXT_LABELS[task]+' | '+str(beta)+'% | '+' | '.join(vals)+' |')
            if ti<2:body.append(r'\midrule')
        body += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        tables.append('\n'.join(body))
        md.append('')
    notes=[
        r'\noindent\textbf{Protocol.} $\beta$ is the fraction of overlapping users held out as target cold-start test users. '
        'We retain all target-only users in training. Music-to-Movie removes cross-domain duplicate items from the source '
        'and contains 17,904 overlapping users (the original paper has 18,031). The other tasks contain 37,388 and 16,738 overlapping users.',
        r'\par\smallskip\noindent\textbf{Backbone and training.} TGTOnly, EMCDR and PFCDR use the official 10-dimensional MF. '
        'All LIST variants receive the same newly trained MF Top-100 IDs/ranks; their established 32-dimensional list/target '
        'representations are retained. Source and target MF train for 10 epochs using Adam (0.01). '
        'Source/target batches are 512 for Book tasks and 256 for Music-to-Movie. '
        'EMCDR uses mapping batch 128 and learning rate 0.01. PFCDR uses full-catalog inversion (300 iterations, batch 200), '
        'mapping learning rate 0.005, mapping batch 512/512/1024, and the released MF implementation\textquotesingle s 70,000-user voting rule.',
        r'\par\smallskip\noindent\textbf{LIST variants.} LIST V1 uses 2,048 interaction batches; LIST V1 + Langevin and LAPT use '
        '64 complete-user batches. Both Langevin variants retain the same fixed sampling settings. LAPT adds the current '
        'rank-aware shared-pattern reconstruction and residual attention with masked-ID supervision. '
        'No SID/coarse codes or additional hyperparameter search are used.',
        r'\par\smallskip\noindent\textbf{Evaluation.} Epochs (and LIST correction weights) are selected on an independent inner '
        'validation split, then models are freshly refit on all outer training users. All 54 prediction arrays are frozen '
        'before test-label evaluation. This differs from the released code\textquotesingle s best-test-epoch reporting. '
        'Results are for one seed, not the paper\textquotesingle s ten-run averages. Bold identifies the best unrounded value within each row. '
        'No statistical-significance claim is made.'
    ]
    document='\n'.join([r'\documentclass[10pt]{article}',r'\usepackage[a4paper,margin=1.4cm]{geometry}',
        r'\usepackage{booktabs,amsmath,textcomp}',r'\usepackage[T1]{fontenc}',r'\pagestyle{empty}',
        r'\begin{document}',r'\begin{center}\large\textbf{Cross-Domain Recommendation under MF: Full Comparison}\end{center}',
        r'\noindent Seed: 2020. Three tasks; $\beta\in\{20\%,50\%,80\%\}$; six methods.',
        *tables,r'\footnotesize',*notes,r'\end{document}',''])
    (root/'main_results.tex').write_text(document)
    (root/'tables_only.tex').write_text('\n\n'.join(tables)+'\n')
    md += ['## 实验口径','',
        '- MF源模型及目标MF均为作者10维、10轮、Adam 0.01；Book任务源/目标batch512，Music→Movie为256。',
        '- EMCDR：10轮映射预算，batch128，lr0.01。PFCDR：完整目录反演300次，batch200，lr1e-4，weight decay1e-4；映射10轮预算、lr0.005，batch512/512/1024；沿用MF实现实际固定的70000投票用户。',
        '- TGTOnly直接使用同一目标MF预训练检查点；三种作者方法均使用原始MF前向，不加入均值回退或评分截断。',
        '- 三种LIST共享同一MF Top100。LIST V1为原始普通均值残差版本，2048交互批；两个郎之万版本均为64完整用户批，保留现有32维编码、hidden64、10轮预算、学习率0.001。',
        '- 郎之万沿用固定20步、训练/预测步长0.25、32/128链、能量上界5；LAPT仅增加此前确认的共享模式重构、残差注意力及遮挡监督。',
        '- β是目标冷用户在重叠用户中的测试比例。源端删除跨域重复物品，Music→Movie重叠人数17904；Book→Movie为37388，Book→Music为16738。',
        '- 采用独立验证选轮次及LIST的λ，然后从头使用全量外层训练集重训；测试标签在全部54组预测冻结后统一读取。',
        '- 这是作者公开默认参数下的单种子复现比较，未复现论文调参搜索或十次均值；验证选模、去重及固定ID排序划分与作者原始执行口径有明确差别。',
        '- LIST V1与郎之万版本的训练单位不同，LAPT也增加参数和辅助目标，因此不能把所有差异归于单一因素。','',
        '[CSV](results.csv) · [完整LaTeX](main_results.tex) · [仅表格LaTeX](tables_only.tex) · [PDF](main_results.pdf)','']
    (root/'report_zh.md').write_text('\n'.join(md))
    compiler=Path('/home/hehao32/.local/share/lapt-tools/tectonic')
    with (root/'latex_build.log').open('w') as log:
        subprocess.run([str(compiler),'--keep-logs','--keep-intermediates','--outdir',str(root),str(root/'main_results.tex')],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    assert (root/'main_results.pdf').read_bytes().startswith(b'%PDF-')
    docs=Path('/home/hehao32/BlackBox_CDR/docs')
    for source,target in [('main_results.tex','lapt_mf_full_seed2020.tex'),
                          ('tables_only.tex','lapt_mf_full_seed2020_tables.tex'),
                          ('main_results.pdf','lapt_mf_full_seed2020.pdf'),
                          ('results.csv','lapt_mf_full_seed2020_results.csv')]:
        shutil.copyfile(root/source,docs/target)
    print('EXPORTED_LATEX_AND_PDF',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    export(p.parse_args().root)
