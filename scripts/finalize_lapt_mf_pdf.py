"""Repair report-only paragraph/quote rendering without changing frozen trainers.

The first exporter remains in the frozen experiment code manifest. This
postprocessor preserves its initial outputs, records all formatting changes,
and checks all 108 rounded metric values against the verified CSV.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0,'/home/hehao32/.local/share/lapt-tools/python')
import pymupdf


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def finalize(root):
    assert json.loads((root/'independent_verification.json').read_text())['status']=='PASS'
    history=root/'report_history';history.mkdir(exist_ok=True)
    for name in ['main_results.tex','main_results.pdf']:
        if not (history/name).exists():shutil.copyfile(root/name,history/name)
    tex=(root/'main_results.tex').read_text()
    # The original normal Python strings converted \t in \textquotesingle
    # into a tab; use ordinary English possessives in the final source.
    tex=tex.replace(chr(9)+'extquotesingle s',"'s")
    tex=tex.replace('six methods.\n','six methods.\\par\\medskip\n')
    tex=tex.replace('\\footnotesize\n\\noindent','\\par\\medskip\\footnotesize\n\\noindent')
    with (root/'results.csv').open() as f:rows=list(csv.DictReader(f))
    if '\\textbf{Mean fallback.}' not in tex:
        counts={m:sum(r['method']==m and r['lambda_value']!='' and float(r['lambda_value'])==0 for r in rows)
                for m in ['LIST_V1','LIST_V1_Langevin','LAPT']}
        note=(r'\par\smallskip\noindent\textbf{Mean fallback.} Validation selects $\lambda=0$ in '
              +str(counts['LAPT'])+' of nine LAPT settings and '+str(counts['LIST_V1_Langevin'])
              +' of nine LIST V1 + Langevin settings; these predictions equal ordinary training-item means. '
              +'LIST V1 selects nonzero corrections in all nine settings.\n')
        tex=tex.replace('\\end{document}',note+'\\end{document}')
    (root/'main_results.tex').write_text(tex)
    with (root/'latex_build_final.log').open('w') as log:
        subprocess.run(['/home/hehao32/.local/share/lapt-tools/tectonic','--keep-logs','--outdir',str(root),
                        str(root/'main_results.tex')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    pdf=pymupdf.open(root/'main_results.pdf')
    text='\n'.join(p.get_text() for p in pdf)
    expected=Counter(f'{float(r[m]):.4f}' for r in rows for m in ['mae','rmse'])
    actual=Counter(re.findall(r'\b\d+\.\d{4}\b',text))
    assert expected==actual and sum(actual.values())==108
    assert 'extquotesingle' not in text
    outside=[]
    for i,page in enumerate(pdf):
        for w in page.get_text('words'):
            if w[0]<0 or w[1]<0 or w[2]>page.rect.width or w[3]>page.rect.height:outside.append([i,w])
        page.get_pixmap(matrix=pymupdf.Matrix(1.6,1.6)).save(root/f'pdf_preview_{i+1}.png')
    assert not outside
    (root/'pdf_text.txt').write_text(text)
    record=dict(status='PASS',pages=len(pdf),metric_cells=108,all_rounded_metrics_match_csv=True,
        text_outside_page=outside,pdf_sha256=sha(root/'main_results.pdf'),tex_sha256=sha(root/'main_results.tex'),
        original_tex_sha256=sha(history/'main_results.tex'),original_pdf_sha256=sha(history/'main_results.pdf'),
        changes=['End subtitle paragraph before tables','Start protocol notes in a new paragraph','Repair possessive apostrophes',
                 'Disclose validation-selected zero-correction cases from the verified CSV'],
        metric_data_unchanged=True,frozen_training_code_unchanged=True,postprocessor_sha256=sha(Path(__file__)))
    (root/'pdf_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    docs=Path('/home/hehao32/BlackBox_CDR/docs')
    shutil.copyfile(root/'main_results.tex',docs/'lapt_mf_full_seed2020.tex')
    shutil.copyfile(root/'main_results.pdf',docs/'lapt_mf_full_seed2020.pdf')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    finalize(p.parse_args().root)
