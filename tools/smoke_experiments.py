"""Bounded all-experiment smoke: run, reload, reuse, batch, analyze; no production study."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.common.workflow import EXPERIMENTS, config, run, batch
from experiments.common.analysis import analyze, load_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);checks=[]
    for experiment in EXPERIMENTS:
        cases=json.loads((Path('experiments')/experiment/'configs/batch.json').read_text())['cases']
        b=batch(experiment,config(experiment),cases,a.output/'runs')
        manifest=json.loads((b/'manifest.json').read_text())
        assert manifest['status']=='completed',manifest
        paths=[Path(m['path']) for m in manifest['members']]
        for path in paths:load_run(path)
        digest={str(path):hashlib.sha256((path/'results.npz').read_bytes()).hexdigest() for path in paths}
        report=analyze(paths,output_root=a.output/'reports',name=experiment)
        again=run(experiment,config(experiment),a.output/'runs')
        assert again.resolve() in [v.resolve() for v in paths] or experiment in ('convergence','sdirk2_transition','mean_reverting','adaptive_tolerance')
        for path in paths:assert digest[str(path)]==hashlib.sha256((path/'results.npz').read_bytes()).hexdigest()
        if experiment in ('convergence','sdirk2_transition','mean_reverting'):
            ref=next(Path(m['path']) for m in manifest['members'] if m['case_id']=='reference')
            analyze([v for v in paths if v!=ref],reference=ref,recipe='errors',name=experiment,output_root=a.output/'reports')
        if experiment=='spatial_spectrum':
            analyze(paths,recipe='spectrum',name=experiment,output_root=a.output/'reports')
            run(experiment,config(experiment,'experiments/spatial_spectrum/configs/cosx.json'),a.output/'runs')
        if experiment=='convergence':
            run(experiment,config(experiment,'experiments/convergence/configs/prescribed.json'),a.output/'runs')
        if experiment=='bursting':
            analyze(paths,recipe='pchip_kde',sample_dt=.0005,name=experiment,output_root=a.output/'reports')
            analyze(paths,recipe='events',sample_dt=.0005,threshold=1.,name=experiment,output_root=a.output/'reports')
        checks.append({'experiment':experiment,'batch':str(b),'report':str(report),'status':'passed'})
        print('PASS',experiment,flush=True)
    (a.output/'smoke-summary.json').write_text(json.dumps(checks,indent=2)+'\n')

if __name__=='__main__':main()
