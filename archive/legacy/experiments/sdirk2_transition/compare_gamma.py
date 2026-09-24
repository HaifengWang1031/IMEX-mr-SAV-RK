"""Read-only comparison of two saved gamma scans with identical physical controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from analyze import load_run
from run import atomic_json


def compare_gamma(base_path,new_path,destination):
    base_path,new_path,destination=map(Path,(base_path,new_path,destination))
    base,new=[json.loads(p.read_text())for p in (base_path,new_path)]
    ca,cb=[{k:v for k,v in s['case'].items()if k!='gamma'}for s in (base,new)]
    if ca!=cb or base['reference_steps']!=new['reference_steps']:
        raise ValueError('Only gamma may change.')
    ra,ma=load_run(base_path.parent,base['reference_id'],base['case'])
    rb,mb=load_run(new_path.parent,new['reference_id'],new['case'])
    if ma['config']['source_sha256']!=mb['config']['source_sha256']:
        raise ValueError('Numerical source versions differ.')
    reference_difference=float(np.linalg.norm(ra['omega']-rb['omega'])/np.linalg.norm(ra['omega']))
    a={r['steps']:r for r in base['records']};b={r['steps']:r for r in new['records']}
    if set(a)!=set(b):
        raise ValueError('Step counts differ; select matching comparisons explicitly.')
    records=[];original_difference=0.
    for n in sorted(a,reverse=True):
        wa,oa=load_run(base_path.parent,a[n]['methods']['IMEX_RK2']['run_id'],base['case'])
        wb,ob=load_run(new_path.parent,b[n]['methods']['IMEX_RK2']['run_id'],new['case'])
        if oa['status']!=ob['status'] or not np.array_equal(wa['times'],wb['times']):
            raise ValueError('Original-method controls do not reproduce.')
        original_difference=max(original_difference,float(np.max(np.abs(wa['omega']-wb['omega']))))
        row={'steps':n,'tau':a[n]['tau']}
        for key,item,path,s in [('original',a[n]['methods']['IMEX_RK2'],base_path,base),
                                ('gamma_base',a[n]['methods']['SDIRK2_mr_SAV'],base_path,base),
                                ('gamma_new',b[n]['methods']['SDIRK2_mr_SAV'],new_path,new)]:
            value,meta=load_run(path.parent,item['run_id'],s['case'])
            row[key]={'status':item['status'],'run_id':item['run_id'],
                      'max_abs_q_minus_one':float(np.max(np.abs(value['diagnostics'][:,1]-1))),
                      'last_state_time':meta['last_state_time'],
                      'error':item['errors'][-1] if item['status']=='completed' else None}
        records.append(row)
    destination.mkdir(parents=True,exist_ok=True)
    audit={'base_comparison':str(base_path),'new_comparison':str(new_path),
           'gamma_base':base['case']['gamma'],'gamma_new':new['case']['gamma'],
           'physical_controls':ca,'reference_relative_difference':reference_difference,
           'original_max_absolute_difference':original_difference,'records':records}
    atomic_json(destination/'gamma_comparison.json',audit)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for key,label,color in [('original','SDIRK2','#D55E00'),
                             ('gamma_base',f"mr-SAV, gamma={base['case']['gamma']:g}",'#0072B2'),
                             ('gamma_new',f"mr-SAV, gamma={new['case']['gamma']:g}",'#009E73')]:
        x=[r['tau']for r in records]
        y=[100*r[key]['error']['omega_relative']if r[key]['error'] else np.nan for r in records]
        axes[0].loglog(x,y,'-o',ms=3,color=color,label=label)
        failed=[r['tau']for r in records if r[key]['error'] is None]
        axes[0].plot(failed,[.98]*len(failed),'x',color=color,transform=axes[0].get_xaxis_transform())
        window=[(xx,yy)for xx,yy in zip(x,y)if .075<=xx<=.086]
        axes[1].plot([p[0]for p in window],[p[1]for p in window],'-o',ms=4,color=color,label=label)
        r=next(r for r in records if r['steps']==74)
        folder=new_path.parent if key=='gamma_new' else base_path.parent
        value,_=load_run(folder,r[key]['run_id'])
        axes[2].plot(value['diagnostics'][:,0],np.abs(value['diagnostics'][:,1]-1),color=color,label=label)
    axes[0].set(xlabel='Fixed step',ylabel='Vorticity relative error (%)',title='T=6: full step range')
    axes[1].set(xlabel='Fixed step',ylabel='Vorticity relative error (%)',title='Transition interval')
    axes[2].set(xlabel='Time',ylabel=r'$|q-1|$',title='Auxiliary variable: step=6/74')
    for ax in axes:
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle(f"N={ca['grid']}, nu={ca['nu']:g}, A={ca['amplitude']:g}, forcing=cos(x); only gamma changes")
    fig.savefig(destination/'gamma_comparison.png',dpi=220)
    fig.savefig(destination/'gamma_comparison.svg');plt.close(fig)
    print(json.dumps(audit,indent=2))
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base',type=Path);p.add_argument('new',type=Path)
    p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args();compare_gamma(args.base,args.new,args.destination)
