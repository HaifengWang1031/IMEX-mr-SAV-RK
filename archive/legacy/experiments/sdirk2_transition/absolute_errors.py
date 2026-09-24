"""Absolute vorticity L2 errors versus fixed step and physical time.

Tables read the saved fixed-step studies. Explicit 'compute' supplements time
histories on common method/reference nodes without interpolation or clipped
steps. 'analyze' only reads saved results; it never invokes integration.
"""
from dataclasses import replace
import argparse
import inspect
import json
from math import gcd
import os
from pathlib import Path
from time import perf_counter
import warnings
import numpy as np
from run import ROOT, Case, METHODS, make_solver, provenance, digest, canonical, atomic_json
from analyze import load_run
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
DATA=ROOT/'data/sdirk2_transition/absolute_errors'
FIG=ROOT/'fig/sdirk2_transition/absolute_errors'
TABLE_COUNTS=[40,56,64,68,72,76,80,84,88,96,128,256,512]
CURVE_COUNTS=[96,72,64,40]
REF_STEPS=1536
COLORS=['#D55E00','#0072B2','#009E73']
LABELS=['SDIRK2','mr-SAV, gamma=5','mr-SAV, gamma=1000']


def inputs():
    rec=json.loads((HERE/'recommended.json').read_text())
    base_path=ROOT/rec['comparison'];base=json.loads(base_path.read_text())
    candidates=[]
    for p in base_path.parent.glob('comparison_*.json'):
        s=json.loads(p.read_text())
        c=dict(s['case']);c['gamma']=base['case']['gamma']
        if c==base['case'] and s['case']['gamma']==1000 and [r['steps']for r in s['records']]==rec['counts']:
            candidates.append((p,s))
    if len(candidates)!=1:
        raise ValueError('Expected one matching gamma=1000 comparison.')
    return rec,base_path,base,*candidates[0]


def capture(case,method,steps,indices,reference=None):
    """Bounded one-step driver, recording only requested aligned nodes.

    Reference fields are saved for reuse. Method runs save absolute errors and
    terminal state, not the full trajectory. Existing computations are reused
    only when case, method, samples, reference and numerical code match.
    """
    DATA.mkdir(parents=True,exist_ok=True)
    prov=provenance()
    cfg={'case':case.__dict__,'method':method,'steps':steps,'sample_indices':indices,
         'reference_id':None if reference is None else json.loads(str(reference['metadata']))['run_id'],
         'source':prov['source_sha256'],'capture_source':inspect.getsource(capture),
         'blowup_bound_factor':100.,'norm':'sqrt(hx*hy*sum((omega-reference)**2))'}
    key=digest(cfg);path=DATA/f'{key}.npz';manifest=DATA/f'{key}.json'
    if path.exists():
        with np.load(path,allow_pickle=False)as f:
            data={k:f[k].copy()for k in f.files}
        if json.loads(str(data['metadata']))['config']!=cfg:
            raise ValueError(f'Configuration mismatch: {path}')
        return data
    meta={'run_id':key,'config':cfg,'provenance':prov,'status':'incomplete'}
    atomic_json(manifest,meta)
    solver=make_solver(case,method);w=solver.Omega0.copy();q=1.
    tau=case.final_time/steps;times=np.array(indices)*tau
    errors=np.full(len(indices),np.nan);qs=np.full(len(indices),np.nan)
    states=[];saved_times=[];lookup={int(i):j for j,i in enumerate(indices)}
    def save(i):
        j=lookup[i];qs[j]=q
        if reference is None:
            states.append(w.copy());saved_times.append(i*tau)
        else:
            matches=np.flatnonzero(np.isclose(reference['times'],i*tau,rtol=0,atol=1e-11))
            if len(matches)!=1:
                raise ValueError(f'No unique reference at physical time {i*tau}.')
            difference=w-reference['omega'][matches[0]]
            errors[j]=np.sqrt(solver.h*np.sum(difference**2))
    save(0);status='completed';message='';last_time=0.;start=perf_counter()
    with warnings.catch_warnings(record=True)as caught:
        for i in range(1,steps+1):
            try:
                with np.errstate(over='raise',invalid='raise',divide='raise'):
                    w,qnew=solver.step(w[None],np.array([q]),(i-1)*tau,np.array([tau]))
                    q=float(qnew);last_time=i*tau
                    if not np.isfinite(w).all()or not np.isfinite(q):
                        status='nonfinite';break
                    if i in lookup:save(i)
                    rms=float(np.sqrt(np.mean(w*w)))
                    if rms>100*(case.amplitude+abs(case.force)*last_time/np.sqrt(2)):
                        status='solution_blowup';break
            except FloatingPointError as exc:
                status='floating_point_error';message=str(exc);break
            except (RuntimeError,ValueError,OverflowError)as exc:
                status='solver_failure';message=str(exc);break
    meta.update(status=status,message=message,last_state_time=last_time,
                elapsed_seconds=perf_counter()-start,warnings=sorted({str(w.message)for w in caught}))
    data={'metadata':np.array(canonical(meta)),'times':times,'errors':errors,'q':qs,'last_state':w}
    if reference is None:
        data['omega']=np.array(states);data['times']=np.array(saved_times)
    tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb')as f:np.savez_compressed(f,**data)
    os.replace(tmp,path);atomic_json(manifest,meta)
    print(method,'gamma=',case.gamma,'steps=',steps,status,f"{meta['elapsed_seconds']:.2f}s",flush=True)
    return data


def compute():
    _,base_path,base,new_path,new=inputs();case=Case(**base['case'])
    # All sample times are genuine nodes of both the method and ETDRK4 grids.
    reference_indices={0,REF_STEPS};samples={}
    for n in CURVE_COUNTS:
        shared=gcd(n,REF_STEPS)
        samples[n]=list(range(0,n+1,n//shared))
        reference_indices.update(range(0,REF_STEPS+1,REF_STEPS//shared))
    ref=capture(case,'ETDRK4',REF_STEPS,sorted(reference_indices))
    if json.loads(str(ref['metadata']))['status']!='completed':
        raise RuntimeError('Reference failed; no error analysis is valid.')
    old_ref,_=load_run(base_path.parent,base['reference_id'])
    np.testing.assert_allclose(ref['last_state'],old_ref['omega'][-1],rtol=0,atol=1e-12)
    index={'reference_id':json.loads(str(ref['metadata']))['run_id'],'curves':[],
           'definition':'absolute vorticity L2 error','reference_terminal_max_difference':float(np.max(np.abs(ref['last_state']-old_ref['omega'][-1])))}
    for n in CURVE_COUNTS:
        runs=[]
        for method,gamma,source,path in [('IMEX_RK2',5.,base,base_path),
                                        ('SDIRK2_mr_SAV',5.,base,base_path),
                                        ('SDIRK2_mr_SAV',1000.,new,new_path)]:
            result=capture(replace(case,gamma=gamma),method,n,samples[n],ref)
            meta=json.loads(str(result['metadata']))
            old=next(r for r in source['records']if r['steps']==n)['methods'][method]
            old_data,old_meta=load_run(path.parent,old['run_id'])
            if meta['status']!=old_meta['status']:
                raise ValueError('Resampled run status differs from saved study.')
            np.testing.assert_allclose(result['last_state'],old_data['last_state'],rtol=1e-8,atol=1e-10)
            if meta['status']=='completed':
                np.testing.assert_allclose(result['errors'][-1],old['errors'][-1]['omega_absolute'],rtol=1e-8,atol=1e-11)
            runs.append({'run_id':meta['run_id'],'method':method,'gamma':gamma,'status':meta['status'],
                         'last_state_time':meta['last_state_time'],'original_run_id':old['run_id'],
                         'terminal_max_difference':float(np.max(np.abs(result['last_state']-old_data['last_state'])))})
        index['curves'].append({'steps':n,'tau':case.final_time/n,'runs':runs})
    atomic_json(DATA/'index.json',index)


def value(row,method,time):
    # Preserve valid snapshots even if the same run fails at a later time.
    values=[e['omega_absolute']for e in row['methods'][method]['errors']if abs(e['time']-time)<1e-11]
    return values[0]if len(values)==1 else np.nan


def sci(x,tex=False):
    if not np.isfinite(x):return r'\texttt{stopped}'if tex else 'stopped'
    if x==0:return '$0$'if tex else '0'
    mantissa,exponent=f'{x:.4e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'if tex else f'{x:.4e}'


def tables(summary,gamma,output=None):
    output=HERE if output is None else Path(output)
    output.mkdir(parents=True,exist_ok=True)
    case=summary['case']
    times=[1.5,3.,4.5,6.]
    rows=[next(r for r in summary['records']if r['steps']==n)for n in TABLE_COUNTS]
    rows.sort(key=lambda r:r['tau'],reverse=True)
    cube=np.array([[[value(r,m,t)for m in METHODS]for t in times]for r in rows])
    taus=np.array([r['tau']for r in rows]);rates=np.full_like(cube,np.nan)
    for i in range(1,len(rows)):
        valid=np.isfinite(cube[i-1])&np.isfinite(cube[i])&(cube[i-1]>0)&(cube[i]>0)
        rates[i][valid]=np.log(cube[i-1][valid]/cube[i][valid])/np.log(taus[i-1]/taus[i])
    stem=f'absolute_errors_gamma{gamma:g}'
    tex=[r'\begin{table}',r'\centering',
         rf'\caption{{Absolute vorticity $L^2$ errors and adjacent-step observed rates for $\gamma={gamma:g}$, $\nu={case['nu']:g}$, $N={case['grid']}$, and forcing ${case['force']:g}\cos({case['force_k']}x)$. ETDRK4 uses $\tau_{{\rm ref}}={case['final_time']:g}/{summary['reference_steps']}$. Rates are raw observed slopes; non-asymptotic values do not imply convergence order. \texttt{{stopped}} denotes a run stopped before the observation time, not a fabricated NaN.}}',
         rf'\label{{tab:absolute-errors-gamma{gamma:g}}}',r'\setlength{\tabcolsep}{3pt}',
         r'\resizebox{\textwidth}{!}{',r'\begin{tabular}{r '+'cc '*8+'}',r'\toprule',
         '& '+' & '.join(rf'\multicolumn{{4}}{{c}}{{$t={t:g}$}}'for t in times)+r'\\',
         ''.join(rf'\cmidrule(lr){{{2+4*j}-{5+4*j}}}'for j in range(4)),
         '& '+' & '.join([r'\multicolumn{2}{c}{SDIRK2} & \multicolumn{2}{c}{SDIRK2-mr-SAV}']*4)+r'\\',
         r'$\tau$ & '+' & '.join(['Error & Rate']*8)+r'\\',r'\midrule']
    md=[f'# gamma={gamma:g}：涡量绝对 L² 误差', '',
        '| tau | '+' | '.join(f't={t:g}: {label}'for t in times for label in ('SDIRK2','mr-SAV'))+' |',
        '|---:|'+'---:|'*8]
    for i,r in enumerate(rows):
        cells=[]
        for j in range(4):
            for k in range(2):
                rate=f'{rates[i,j,k]:.2f}'if np.isfinite(rates[i,j,k])else '--'
                cells.extend([sci(cube[i,j,k],True),rate])
        tex.append(f"{r['tau']:.9f} & "+' & '.join(cells)+r' \\')
        md.append(f"| {r['tau']:.9f} | "+' | '.join(sci(x)for x in cube[i].ravel())+' |')
    tex.extend([r'\bottomrule',r'\end{tabular}}',r'\end{table}'])
    (output/f'{stem}.tex').write_text('\n'.join(tex)+'\n')
    (output/f'{stem}.md').write_text('\n'.join(md)+'\n')
    return taus,cube


def analyze():
    _,_,base,_,new=inputs();FIG.mkdir(parents=True,exist_ok=True)
    # Both table families use exactly the absolute norm from the original file.
    ta,a=tables(base,5);tb,b=tables(new,1000)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for j,(ax,t)in enumerate(zip(axes.flat,[1.5,3.,4.5,6.])):
        for k,series in enumerate([a[:,j,0],a[:,j,1],b[:,j,1]]):
            ax.loglog(ta,series,'-o',ms=4,color=COLORS[k],label=LABELS[k])
            missing=ta[~np.isfinite(series)]
            ax.plot(missing,[.98-k*.04]*len(missing),'x',color=COLORS[k],transform=ax.get_xaxis_transform())
        ax.set(title=f't={t:g}',xlabel=r'Fixed step $\tau$',ylabel=r'Absolute $L^2$ error')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Vorticity error versus step: nu=0.2, N=256, forcing=cos(x)')
    fig.savefig(FIG/'error_vs_tau.png',dpi=220);fig.savefig(FIG/'error_vs_tau.svg');plt.close(fig)
    if not(DATA/'index.json').exists():
        raise FileNotFoundError('Missing time histories; run absolute_errors.py compute explicitly.')
    index=json.loads((DATA/'index.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    report=[]
    for ax,curve in zip(axes.flat,index['curves']):
        for k,item in enumerate(curve['runs']):
            result,meta=load_run(DATA,item['run_id'])
            valid=np.isfinite(result['errors'])&(result['errors']>0)
            ax.semilogy(result['times'][valid],result['errors'][valid],'-o',ms=3,color=COLORS[k],label=LABELS[k])
            if meta['status']!='completed':
                ax.axvline(meta['last_state_time'],color=COLORS[k],ls=':',lw=1)
                ax.text(meta['last_state_time'],.96,'stopped',rotation=90,va='top',ha='right',
                        transform=ax.get_xaxis_transform(),color=COLORS[k],fontsize=8)
            report.append({'tau':curve['tau'],'method':item['method'],'gamma':item['gamma'],
                           'time':result['times'].tolist(),
                           'absolute_error':[float(x)if np.isfinite(x)else None for x in result['errors']],
                           'status':meta['status'],'stop_time':meta['last_state_time']})
        ax.set(title=rf"$\tau={curve['tau']:.8g}$",xlabel='Physical time',ylabel=r'Absolute $L^2$ error',xlim=(0,6.1))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Vorticity error versus time: exact common time nodes, no interpolation')
    fig.savefig(FIG/'error_vs_time.png',dpi=220);fig.savefig(FIG/'error_vs_time.svg');plt.close(fig)
    atomic_json(DATA/'absolute_time_errors.json',report)
    time_rows=[r for r in report if abs(r['tau']-1/12)<1e-12]
    text=['# 固定 tau=1/12：绝对 L² 误差随时间变化','',
          '| t | SDIRK2 | mr-SAV gamma=5 | mr-SAV gamma=1000 |','|---:|---:|---:|---:|']
    for t in [1.5,3.,4.5,4.75,5.,5.25,5.5,5.75,6.]:
        values=[r['absolute_error'][r['time'].index(t)]for r in time_rows]
        text.append(f'| {t:g} | '+' | '.join(f'{v:.6e}'if v is not None else 'stopped'for v in values)+' |')
    (HERE/'absolute_errors_time.md').write_text('\n'.join(text)+'\n')
    # Dense terminal transition table includes odd counts without missing times.
    lines=['# T=6：过渡区绝对 L² 误差','','| tau | SDIRK2 | gamma=5 | gamma=1000 |','|---:|---:|---:|---:|']
    for n in [80,78,77,76,75,74,73,72,71,70,68,64]:
        ra=next(r for r in base['records']if r['steps']==n);rb=next(r for r in new['records']if r['steps']==n)
        lines.append(f"| {ra['tau']:.9f} | "+' | '.join(sci(v)for v in [value(ra,METHODS[0],6),value(ra,METHODS[1],6),value(rb,METHODS[1],6)])+' |')
    (HERE/'absolute_errors_transition.md').write_text('\n'.join(lines)+'\n')
    print('Tables:',HERE/'absolute_errors_gamma5.tex',HERE/'absolute_errors_gamma1000.tex')
    print('Figures:',FIG)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['compute','analyze'])
    args=parser.parse_args()
    if args.command=='compute':compute()
    else:analyze()
