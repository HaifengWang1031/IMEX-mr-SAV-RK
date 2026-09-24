"""Read-only reference-time and spatial-refinement comparisons of saved runs.

For nested periodic grids, compare fields at common collocation points (no
interpolation). Report each physical time, not only a single terminal value.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze import load_run


def relative(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def validate(coarse_path, fine_path, destination):
    coarse_path,fine_path=Path(coarse_path),Path(fine_path)
    a,ma=load_run(coarse_path.parent,coarse_path.stem)
    b,mb=load_run(fine_path.parent,fine_path.stem)
    ca,cb=ma['config'],mb['config']
    if ma['status']!='completed' or mb['status']!='completed':
        raise ValueError('Validation requires completed runs.')
    if ca['method']!=cb['method'] or ca['source_sha256']!=cb['source_sha256']:
        raise ValueError('Different methods/source versions.')
    changes={k:(ca['case'][k],cb['case'][k])for k in ca['case']if ca['case'][k]!=cb['case'][k]}
    if set(changes)-{'grid'}:
        raise ValueError(f'Physical/algorithm configuration changed: {changes}')
    if not np.array_equal(a['times'],b['times']):
        raise ValueError('Physical times differ.')
    na,nb=ca['case']['grid'],cb['case']['grid']
    if nb<na or nb%na:
        raise ValueError('Require nested grids with fine >= coarse.')
    if na!=nb and ca['steps']!=cb['steps']:
        raise ValueError('Spatial comparison must use the same time step.')
    if na==nb and cb['steps']!=2*ca['steps']:
        raise ValueError('Time validation requires exactly halving the step.')
    stride=nb//na
    result={'coarse_id':ma['run_id'],'fine_id':mb['run_id'],
            'kind':'reference_step_halving' if na==nb else 'spatial_refinement',
            'coarse_config':ca,'fine_config':cb,'errors':[]}
    for t,wa,wb in zip(a['times'],a['omega'],b['omega']):
        result['errors'].append({'time':float(t),'omega_relative':relative(wa,wb[::stride,::stride])})
    Path(destination).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['errors']))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('coarse',type=Path);p.add_argument('fine',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();validate(args.coarse,args.fine,args.output)
