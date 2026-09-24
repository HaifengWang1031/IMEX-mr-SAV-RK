"""Rebuild absolute-error tables for saved gamma=5,100,500,1000 scans.

Read-only with respect to computation data. Missing/incompatible scans raise an
error; no numerical integration is launched by this script.
"""
import json
from pathlib import Path
import numpy as np
from run import ROOT, atomic_json
from analyze import load_run
from absolute_errors import HERE, TABLE_COUNTS, tables, value, sci

GAMMAS=[5,100,500,1000]
TIMES=[1.5,3.,4.5,6.]


def main():
    rec=json.loads((HERE/'recommended.json').read_text())
    base_path=ROOT/rec['comparison'];base=json.loads(base_path.read_text())
    base_ref,base_meta=load_run(base_path.parent,base['reference_id'],base['case'])
    physical={k:v for k,v in base['case'].items()if k!='gamma'}
    summaries={};sources={};checks={}
    for gamma in GAMMAS:
        matches=[]
        for p in base_path.parent.glob('comparison_*.json'):
            s=json.loads(p.read_text())
            c={k:v for k,v in s['case'].items()if k!='gamma'}
            if c==physical and s['case']['gamma']==gamma and [r['steps']for r in s['records']]==rec['counts']:
                matches.append((p,s))
        if len(matches)!=1:
            raise ValueError(f'Expected one compatible gamma={gamma} scan, found {len(matches)}.')
        p,s=matches[0]
        ref,meta=load_run(p.parent,s['reference_id'],s['case'])
        if meta['status']!='completed' or meta['config']['source_sha256']!=base_meta['config']['source_sha256']:
            raise ValueError('Reference incomplete or numerical source mismatch.')
        np.testing.assert_array_equal(ref['times'],base_ref['times'])
        np.testing.assert_array_equal(ref['omega'],base_ref['omega'])
        for r,br in zip(s['records'],base['records']):
            a,b=r['methods']['IMEX_RK2'],br['methods']['IMEX_RK2']
            if a['status']!=b['status'] or a['errors']!=b['errors']:
                raise ValueError('Original-method control did not reproduce exactly.')
        tables(s,gamma)
        summaries[gamma]={r['steps']:r for r in s['records']}
        sources[str(gamma)]=str(p.relative_to(ROOT))
        checks[str(gamma)]={'reference_snapshots_identical':True,'original_errors_and_statuses_identical':True}
    head='| tau | SDIRK2 | '+ ' | '.join(f'mr-SAV gamma={g}'for g in GAMMAS)+' |'
    sep='|---:|'+'---:|'*5
    text=['# 四个 gamma 的绝对 L² 误差对比','',
          '固定 nu=0.2、N=256、T=6、原初值和非零外力 cos(x)，仅改变 gamma。以下全部为涡量绝对 L² 误差，没有除以参考解范数，也没有转换为百分数。',
          '', '## 固定物理时间，比较不同步长','']
    data=[]
    for t in TIMES:
        text.extend([f'### t={t:g}','',head,sep])
        for n in sorted(TABLE_COUNTS,reverse=True):
            row=summaries[5][n]
            errors=[value(row,'IMEX_RK2',t)]+[value(summaries[g][n],'SDIRK2_mr_SAV',t)for g in GAMMAS]
            text.append(f"| {row['tau']:.9f} | "+' | '.join(sci(e)for e in errors)+' |')
            data.append({'time':t,'steps':n,'tau':row['tau'],'absolute_errors':
                         {k:float(e)if np.isfinite(e)else None for k,e in zip(['SDIRK2',*[f'gamma{g}'for g in GAMMAS]],errors)}})
        text.append('')
    text.extend(['## T=6 的密集过渡步长','',head,sep])
    for n in [80,78,77,76,75,74,73,72,71,70,68,64]:
        r=summaries[5][n]
        errors=[value(r,'IMEX_RK2',6)]+[value(summaries[g][n],'SDIRK2_mr_SAV',6)for g in GAMMAS]
        text.append(f"| {r['tau']:.9f} | "+' | '.join(sci(e)for e in errors)+' |')
    text+=['', '## 说明','',
           '- `stopped` 表示该方法已在该观测时间之前触发预定停止条件；停止前的有效误差仍保留。',
           '- 两个新增 Error/Rate 表保持与 gamma=5、1000 表相同的 13 个步长和四个物理时间。',
           '- 四组参考快照逐点相同，原 SDIRK2 的误差与状态也逐项相同；控制变量核对通过。',
           '- gamma=100、500 本次在 256² 网格补算；没有为它们另做空间加密，也没有补算更密的时间曲线。',
           '- 若误差随 gamma 出现局部非单调，保留实际值；不据此声称某个 gamma 全局最优。',
           '', '运行 `python experiments/sdirk2_transition/gamma_tables.py` 可从持久化结果重建全部四张 LaTeX/Markdown 表及此汇总，不会重新积分。','']
    (HERE/'absolute_errors_gamma_comparison.md').write_text('\n'.join(text))
    atomic_json(HERE/'absolute_errors_gamma_comparison.json',{'physical_controls':physical,'gammas':GAMMAS,
                 'times':TIMES,'sources':sources,'checks':checks,'rows':data})
    print('Generated four gamma tables and absolute_errors_gamma_comparison.md')
    for n in [75,74,73,72,40]:
        print(n,[value(summaries[g][n],'SDIRK2_mr_SAV',6)for g in GAMMAS])


if __name__=='__main__':
    main()
