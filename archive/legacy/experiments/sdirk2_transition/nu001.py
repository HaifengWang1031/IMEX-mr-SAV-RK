"""Explicit nu=0.01 study; compute persists evidence, analyze only reads it.

Controls: original trig initial field, RMS=1, cos(x) forcing, T=6, N=256.
Vary fixed step and gamma. Compare absolute vorticity L2 errors at common
quarter times. Smaller viscosity does not imply a guaranteed SAV advantage.
Runs restart from zero if interrupted; completed compatible NPZs are reused.
"""
import argparse
from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import threading
import time
import numpy as np
from run import ROOT, Case, run, relative_errors, atomic_json
import absolute_errors as ae

OUT=ROOT/'data/sdirk2_transition/nu001'
REPORT=ROOT/'experiments/sdirk2_transition/nu001_results'
CONFIG={
 'case':asdict(Case(grid=256,nu=.01,gamma=5,final_time=6)),
 'gammas':[5,100,500,1000],
 'counts':[40,56,64,68,72,76,80,84,88,96,128,256,512,1024,2048],
 'reference_steps':3072,'fine_reference_steps':6144,'fine_grid':512,
 'log_interval_seconds':30,
}

def meta(a): return json.loads(str(a['metadata']))

def checked_reference(c,n):
    logging.info('Reference N=%s steps=%s',c.grid,n)
    a=run(c,'ETDRK4',n,OUT)
    if meta(a)['status']!='completed':
        raise RuntimeError('Reference failed; no comparison is valid.')
    return a

def compute():
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'config.json',CONFIG)
    atomic_json(OUT/'status.json',{'status':'running','started':time.time()})
    stop=threading.Event()
    started=time.monotonic()
    def heartbeat():
        while not stop.wait(CONFIG['log_interval_seconds']):
            logging.info('RUNNING elapsed=%.1fs; current integration has not returned',time.monotonic()-started)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        c=Case(**CONFIG['case'])
        ref=checked_reference(c,CONFIG['reference_steps'])
        fine=checked_reference(c,CONFIG['fine_reference_steps'])
        space=checked_reference(replace(c,grid=CONFIG['fine_grid']),CONFIG['fine_reference_steps'])
        np.testing.assert_allclose(ref['times'],fine['times'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(space['times'],fine['times'],rtol=0,atol=1e-12)
        h=(2*np.pi/c.grid)**2
        time_error=np.sqrt(h*np.sum((ref['omega']-fine['omega'])**2,axis=(1,2)))
        space_error=np.sqrt(h*np.sum((space['omega'][:,::2,::2]-fine['omega'])**2,axis=(1,2)))
        validation={'times':fine['times'].tolist(),'time_halving_absolute':time_error.tolist(),
                    'spatial_refinement_absolute':space_error.tolist(),
                    'reference_ids':[meta(a)['run_id']for a in (ref,fine,space)],
                    'note':'Errors use the N=256 fine-time reference; refinement differences are empirical uncertainty indicators.'}
        atomic_json(OUT/'validation.json',validation)
        logging.info('Reference validation: temporal max=%g; spatial max=%g',max(time_error),max(space_error))
        originals={}
        for n in CONFIG['counts']:
            logging.info('SDIRK2 steps=%s tau=%g',n,c.final_time/n)
            a=run(c,'IMEX_RK2',n,OUT)
            originals[n]={'run_id':meta(a)['run_id'],'status':meta(a)['status'],'errors':relative_errors(c,a,fine)}
        for gamma in CONFIG['gammas']:
            cg=replace(c,gamma=gamma);records=[]
            for n in CONFIG['counts']:
                logging.info('mr-SAV gamma=%s steps=%s tau=%g',gamma,n,c.final_time/n)
                a=run(cg,'SDIRK2_mr_SAV',n,OUT)
                records.append({'steps':n,'tau':c.final_time/n,'methods':{
                  'IMEX_RK2':originals[n],
                  'SDIRK2_mr_SAV':{'run_id':meta(a)['run_id'],'status':meta(a)['status'],'errors':relative_errors(cg,a,fine)}}})
            atomic_json(OUT/f'comparison_gamma{gamma}.json',{'case':asdict(cg),'reference_id':meta(fine)['run_id'],
                  'reference_steps':CONFIG['fine_reference_steps'],'records':records,'validation':validation})
        analyze()
        atomic_json(OUT/'status.json',{'status':'completed','elapsed_seconds':time.monotonic()-started,'reports':str(REPORT)})
        logging.info('COMPLETED; data=%s reports=%s',OUT,REPORT)
    except Exception as exc:
        atomic_json(OUT/'status.json',{'status':'failed','error':repr(exc)})
        logging.exception('FAILED')
        raise
    finally: stop.set()

def analyze():
    REPORT.mkdir(parents=True,exist_ok=True)
    summaries={g:json.loads((OUT/f'comparison_gamma{g}.json').read_text())for g in CONFIG['gammas']}
    ae.TABLE_COUNTS=CONFIG['counts']
    for g,s in summaries.items(): ae.tables(s,g,REPORT)
    validation=json.loads((OUT/'validation.json').read_text())
    lines=['# nu=0.01：绝对涡量 L² 误差','',
      '固定 N=256、T=6、初值 RMS=1、外力 cos(x)。Error 未除以参考解范数。',
      '参考解为 N=256、6144 步 ETDRK4；下列加密差异是误差可信程度的检查，不能直接作为严格误差界。','',
      '| t | 时间减半差异 | N=256 到 512 差异 |','|---:|---:|---:|']
    for t,a,b in zip(validation['times'],validation['time_halving_absolute'],validation['spatial_refinement_absolute']):
        lines.append(f'| {t:g} | {a:.6e} | {b:.6e} |')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,t in zip(axes.flat,[1.5,3,4.5,6]):
        lines+=['',f'## t={t:g}','','| tau | SDIRK2 | gamma=5 | gamma=100 | gamma=500 | gamma=1000 |','|---:|---:|---:|---:|---:|---:|']
        grid=[];values=[]
        for i,r in enumerate(summaries[5]['records']):
            errors=[ae.value(r,'IMEX_RK2',t)]+[ae.value(summaries[g]['records'][i],'SDIRK2_mr_SAV',t)for g in CONFIG['gammas']]
            grid.append(r['tau']);values.append(errors)
            lines.append(f"| {r['tau']:.9f} | "+' | '.join(ae.sci(e)for e in errors)+' |')
        for j,label in enumerate(['SDIRK2']+[f'mr-SAV gamma={g}'for g in CONFIG['gammas']]):
            ax.loglog(grid,np.array(values)[:,j],'-o',ms=3,label=label)
        ax.set(title=f't={t:g}',xlabel='tau',ylabel='Absolute vorticity L2 error');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(REPORT/'error_vs_tau.png',dpi=180);plt.close(fig)
    lines+=['','stopped：在该观测时间前已停止；停止前的有效误差保留。',
            '图中缺失点对应表中 stopped；非渐近区域的 Rate 不代表收敛阶。',
            '本轮时间比较使用 t=1.5,3,4.5,6 的真实节点，未插值；未生成更密时间轨迹。']
    (REPORT/'comparison.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['compute','analyze']);args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.command=='compute': compute()
    else: analyze()
