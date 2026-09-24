"""Rebuild the recommended-data index and validation audit from saved runs only."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from run import ROOT, Case, METHODS, atomic_json
from analyze import load_run
from validate import validate

HERE=Path(__file__).resolve().parent
FOLDER=ROOT/'data/sdirk2_transition/validated'
COUNTS=[40,56,64,68,70,71,72,73,74,75,76,77,78,79,80,82,84,88,96,128,256,512]


def main():
    comparisons=[]
    for p in FOLDER.glob('comparison_*.json'):
        value=json.loads(p.read_text())
        if [r['steps']for r in value['records']]==COUNTS and value['reference_steps']==1536:
            comparisons.append((p,value))
    def summary(grid):
        case=asdict(Case(grid=grid,nu=.2,gamma=5.,final_time=6.))
        matches=[(p,s)for p,s in comparisons if s['case']==case]
        if len(matches)!=1:
            raise ValueError(f'Expected one final comparison for N={grid}, found {len(matches)}')
        return matches[0]
    p128,s128=summary(128);p256,s256=summary(256)
    metas=[]
    for p in FOLDER.glob('*.npz'):
        _,m=load_run(FOLDER,p.stem)
        metas.append(m)
    def find(grid,method,n,root='legacy'):
        case=asdict(Case(grid=grid,nu=.2,gamma=5.,final_time=6.,root_selection=root))
        matching=[m for m in metas if m['config']['case']==case and m['config']['method']==method
                  and m['config']['steps']==n and m['status']=='completed']
        if len(matching)!=1:
            raise ValueError(f'Expected one compatible completed run: {grid}, {method}, {n}, {root}')
        return matching[0]
    audit={'case':s256['case'],'reference_time':{},'spatial':{},'root_sensitivity':{},'small_step_orders':{}}
    audit_folder=FOLDER/'audit';audit_folder.mkdir(exist_ok=True)
    for grid in (128,256):
        a,b=find(grid,'ETDRK4',768),find(grid,'ETDRK4',1536)
        result=validate(FOLDER/(a['run_id']+'.npz'),FOLDER/(b['run_id']+'.npz'),audit_folder/f'reference_time_{grid}.json')
        audit['reference_time'][str(grid)]=result
    a,b=find(128,'ETDRK4',1536),find(256,'ETDRK4',1536)
    audit['spatial']['reference']=validate(FOLDER/(a['run_id']+'.npz'),FOLDER/(b['run_id']+'.npz'),audit_folder/'reference_space.json')
    for method in METHODS:
        for n in (75,74,73,72):
            a,b=find(128,method,n),find(256,method,n)
            key=f'{method}_{n}'
            audit['spatial'][key]=validate(FOLDER/(a['run_id']+'.npz'),FOLDER/(b['run_id']+'.npz'),audit_folder/f'{key}_space.json')
    for n in (75,74,72,40):
        a,b=find(256,'SDIRK2_mr_SAV',n),find(256,'SDIRK2_mr_SAV',n,'nearest')
        wa,_=load_run(FOLDER,a['run_id']);wb,_=load_run(FOLDER,b['run_id'])
        audit['root_sensitivity'][str(n)]={'legacy_id':a['run_id'],'nearest_id':b['run_id'],
             'terminal_relative_difference':float(np.linalg.norm(wa['omega'][-1]-wb['omega'][-1])/np.linalg.norm(wa['omega'][-1]))}
    for method in METHODS:
        rows=[next(r for r in s256['records']if r['steps']==n)for n in (128,256,512)]
        errors=[r['methods'][method]['errors'][-1]['omega_relative']for r in rows]
        audit['small_step_orders'][method]={'steps':[128,256,512],'errors':errors,
                                           'orders':list(np.log2(np.array(errors[:-1])/errors[1:]))}
    selected=next(r for r in s256['records']if r['steps']==72)
    audit['sampled_time_errors_at_tau_1_over_12']=selected
    for n in (40,56):
        row=next(r for r in s256['records']if r['steps']==n)
        audit[f'large_step_{n}']={}
        for method in METHODS:
            arrays,meta=load_run(FOLDER,row['methods'][method]['run_id'])
            d=arrays['diagnostics']
            audit[f'large_step_{n}'][method]={'run_id':meta['run_id'],'status':meta['status'],
                 'last_state_time':meta['last_state_time'], 'max_rms_over_forced_bound':float(np.max(d[:,2]/d[:,-1])),
                 'max_abs_q_minus_one':float(np.max(np.abs(d[:,1]-1))),
                 'final_error':row['methods'][method]['errors'][-1] if meta['status']=='completed' else None}
    atomic_json(HERE/'validation.json',audit)
    atomic_json(HERE/'recommended.json',{'case':s256['case'],'counts':COUNTS,'selected_steps':74,
                  'comparison':str(p256.relative_to(ROOT)),'comparison_128':str(p128.relative_to(ROOT)),
                  'reference_steps':1536,'validation':'experiments/sdirk2_transition/validation.json'})
    print('Recommended comparison:',p256)
    print('Validation audit:',HERE/'validation.json')


if __name__=='__main__':
    main()
