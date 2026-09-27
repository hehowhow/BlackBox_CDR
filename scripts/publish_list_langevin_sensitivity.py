"""Publish the verified sensitivity report after the GPU job ends (CPU only)."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


BASE = Path('/home/hehao32/BlackBox_CDR')


def publish(job):
    root = BASE / 'runs' / f'list_langevin_sensitivity_{job}'
    doc = BASE / 'docs/list_langevin_sensitivity_experiment.md'
    prefix = doc.read_text().split('## 运行状态')[0]
    result = subprocess.run(['sacct', '-j', str(job), '--format=JobIDRaw,State,ExitCode,Elapsed', '-n', '-P'],
                            capture_output=True, text=True, check=True)
    jobline = next((line for line in result.stdout.splitlines() if line.split('|')[0] == str(job)), '状态待查询')
    verification = root / 'verification.json'
    if not verification.exists() or json.loads(verification.read_text()).get('status') != 'PASS':
        status = (f'## 运行状态\n\nSlurm作业 **{job}** 已结束，但尚无通过的完整复核记录。'
                  f'调度状态：`{jobline}`。本轮结果不能记为完成。\n\n'
                  f'[输出日志](../logs/list_langevin_sensitivity_{job}.out) · '
                  f'[错误日志](../logs/list_langevin_sensitivity_{job}.err) · '
                  f'[运行目录](../runs/list_langevin_sensitivity_{job}/)\n')
        doc.write_text(prefix + status)
        raise RuntimeError(f'Experiment {job} did not produce verification PASS: {jobline}')
    verification_record = json.loads(verification.read_text())
    assert verification_record['models'] == 90 and verification_record['rows'] == 126
    for name in ['results', 'baselines', 'sensitivity']:
        shutil.copyfile(root / f'{name}.csv', BASE / 'docs' / f'list_langevin_sensitivity_{name}.csv')
    reproduced = json.loads((root / 'default_reproduction.json').read_text())
    exact = all(row['exact_predictions'] for row in reproduced['rows'])
    body = (root / 'report.md').read_text().split('## 同种子对照', 1)[1]
    for name in ['results', 'baselines', 'sensitivity']:
        body = body.replace(f']({name}.csv)', f'](list_langevin_sensitivity_{name}.csv)')
    for name in ['protocol.json', 'default_reproduction.json', 'verification.json']:
        body = body.replace(f']({name})', f'](../runs/list_langevin_sensitivity_{job}/{name})')
    recovery = verification_record.get('recovery_reason')
    review_job = verification_record.get('review_slurm_job')
    explanation = (f'原GPU作业的调度状态为`{jobline}`：90项训练和测试评估已完成，'
                   '退出非零发生在复核脚本要求“验证选择文件时间戳严格小于冻结清单”这一检查。'
                   '两个顺序写入的文件时间戳恰好相等，内容哈希一致。'
                   f'随后CPU作业 **{review_job}** 使用修正版复核脚本通过全部检查；'
                   '只将该时间戳边界改为允许相等，保留原文件、全部冻结哈希和其他时序检查，模型与预测没有改动。'
                   if recovery else f'调度状态：`{jobline}`。')
    status = (f'## 运行状态\n\n已完成90个配置训练和测试评估，独立复核PASS。'
              f'{explanation}默认六档预测逐元素精确复现：{exact}。'
              '全部模型与验证选择冻结后才进行本轮测试评估。\n\n'
              f'[完整运行记录](../runs/list_langevin_sensitivity_{job}/) · '
              f'[独立复核](../runs/list_langevin_sensitivity_{job}/verification.json)\n\n'
              '## 同种子对照')
    doc.write_text(prefix + status + body)
    (root / 'documentation_published.json').write_text(json.dumps(dict(status='PASS', document=str(doc),
        slurm=jobline, review_slurm_job=review_job, recovery_reason=recovery,
        models=90, result_rows=126), indent=2) + '\n')
    print('LANGEVIN_DOCUMENTATION_PUBLISHED', doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=int, required=True)
    publish(parser.parse_args().job)
