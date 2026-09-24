"""Read explicit run records; never import an experiment's computation entry point."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def load_run(directory, *, allow_failed=False):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    config=json.loads((directory/'config.json').read_text())
    config=config.get('parameters',config)
    if (manifest['status']!='completed' and not (allow_failed and manifest['status']=='failed')) or manifest['run_id']!=directory.name:
        raise ValueError(f'Not a completed matching run: {directory}')
    path=directory/'results.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            info=json.loads(str(f['metadata']))
            if info['schema']!=1 or (info['stats']['status']!='completed' and not allow_failed):raise ValueError('Unsupported/incomplete result')
            if info['metadata']['identity']!=manifest['identity'] or info['metadata']['run_id']!=manifest['run_id']:
                raise ValueError('Result identity mismatch')
            times=f['times'].copy()
            diagnostics={n:f[f'diagnostics_{i}'].copy() for i,n in enumerate(info['diagnostics'])}
            aux={n:f[f'auxiliary_{i}'].copy() for i,n in enumerate(info['auxiliary'])}
            final=f['final_0'].copy()
            field_names=info['fields']
            if field_names!=['omega']:raise ValueError('Analysis expects one omega field')
        energy=diagnostics['energy'];enstrophy=diagnostics['enstrophy']
    else:
        import h5py
        path=directory/'results.h5'
        with h5py.File(path,'r') as f:
            if f.attrs['status']!='completed' or f.attrs['run_id']!=directory.name or json.loads(f.attrs['identity'])!=manifest['identity']:
                raise ValueError('HDF5 identity mismatch')
            times=f['tn'][:];energy=f['Energy'][:];enstrophy=f['Enstrophy'][:]
            aux={'q':f['q'][:]} if 'q' in f else {}
            final=f['Omega'][-1]
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid time axis')
    for a in [energy,enstrophy,*aux.values()]:
        if a.shape!=times.shape or not np.isfinite(a).all():raise ValueError('Invalid diagnostic trajectory')
    if not np.isfinite(final).all():raise ValueError('Nonfinite final field')
    if manifest['status']=='completed' and not np.isclose(times[-1],config['T'],rtol=1e-12,atol=1e-14):raise ValueError('Endpoint mismatch')
    expected=manifest['identity'].get('parameters')
    if expected is not None and any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Configuration identity mismatch')
    return {'path':directory,'manifest':manifest,'config':config,'time':times,'energy':energy,'enstrophy':enstrophy,'aux':aux,'final':final}


def analyze(paths, *, output_root=None, name='comparison', recipe='diagnostics', reference=None,
            window=None, sample_dt=None, threshold=None, allow_failed=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator
    from scipy.stats import gaussian_kde
    if not name or Path(name).name!=name:raise ValueError('Analysis name must be a single directory name')
    if recipe not in ('diagnostics','errors','spectrum','pchip_kde','events'):raise ValueError('Unknown analysis recipe')
    directory=Path(output_root or ROOT/'reports')/name/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True);(directory/'figures').mkdir();(directory/'tables').mkdir()
    record={'status':'running','inputs':[str(Path(p).resolve()) for p in paths],
            'parameters':{'recipe':recipe,'reference':str(Path(reference).resolve()) if reference else None,
                          'window':window,'sample_dt':sample_dt,'threshold':threshold,'allow_failed':allow_failed},
            'definition':'energy=0.5*integral(|u|^2); enstrophy=0.5*integral(omega^2); actual integration times',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__}}
    def save():
        (directory/'analysis.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if not paths:raise ValueError('Select at least one explicit run')
        runs=[load_run(p,allow_failed=allow_failed) for p in paths];ref=load_run(reference) if reference else None
        record['inputs']=[{'path':str(r['path']),'run_id':r['manifest']['run_id'],'identity':r['manifest']['identity']} for r in runs]
        if ref:record['reference']={'path':str(ref['path']),'identity':ref['manifest']['identity']}
        fig,axes=plt.subplots(2,1,figsize=(8,6),layout='constrained');rows=[]
        detail,ax=plt.subplots(figsize=(7,4),layout='constrained') if recipe!='diagnostics' else (None,None)
        for r in runs:
            c=r['config'];t=r['time'];z=r['enstrophy'];label=f"{c.get('scheme',c.get('M','method'))} N={c['N']} dt={c.get('dt',c.get('tau','adaptive'))} [{r['path'].name[-8:]}]"
            label += ' ['+r['manifest']['status']+']'
            axes[0].plot(t,z,label=label);axes[1].plot(t,r['energy'],label=label)
            row={'run_id':r['path'].name,'status':r['manifest']['status'],'scheme':c.get('scheme',c.get('M')),'N':c['N'],'T':float(t[-1]),'steps':len(t)-1,
                 'final_enstrophy':float(z[-1]),'final_energy':float(r['energy'][-1])}
            if recipe=='errors':
                if r['manifest']['status']!='completed':
                    row.update(omega_l2_error=None,omega_relative_error=None)
                    rows.append(row)
                    continue
                if ref is None:raise ValueError('Error analysis requires an explicit reference run')
                physical=('N','nu','domain','forcing','initial','T','warmup_time','warmup_dt')
                if any(c.get(k)!=ref['config'].get(k) for k in physical) or r['final'].shape!=ref['final'].shape:
                    raise ValueError('Reference model/grid/initial/time mismatch')
                domain=c['domain'];area=(domain[2]-domain[0])*(domain[3]-domain[1])
                err=np.sqrt(area*np.mean((r['final']-ref['final'])**2));norm=np.sqrt(area*np.mean(ref['final']**2))
                row.update(omega_l2_error=float(err),omega_relative_error=float(err/norm) if norm else None)
                ax.scatter(c['dt'] if c['mode']=='fixed' else np.max(np.diff(t)),err,label=label)
                ax.set(xlabel='Maximum time step',ylabel='Final vorticity L2 error',xscale='log',yscale='log')
            elif recipe=='spectrum':
                w=r['final'];n=w.shape[0];hat=np.fft.fft2(w)/n**2;k=np.fft.fftfreq(n)*n
                rings=np.rint(np.hypot(k[:,None],k[None,:])).astype(int)
                spectrum=np.bincount(rings.ravel(),weights=(.5*abs(hat)**2).ravel())
                ax.semilogy(np.arange(len(spectrum)),np.maximum(spectrum,1e-30),label=label)
                ax.set(xlabel='Fourier shell',ylabel='Enstrophy spectrum (spatial mean)')
            elif recipe in ('pchip_kde','events'):
                lo,hi=window or (float(t[0]),float(t[-1]))
                if not t[0]<=lo<hi<=t[-1]:raise ValueError('Window outside saved time range')
                if sample_dt is None or sample_dt<=0:raise ValueError('Specify positive sample_dt for uniform-time statistics')
                grid=lo+sample_dt*np.arange(int(np.floor((hi-lo)/sample_dt))+1)
                values=PchipInterpolator(t,z)(grid)
                if len(values)<3:raise ValueError('At least three uniform samples required')
                if recipe=='pchip_kde':
                    if np.ptp(values)<=np.finfo(float).eps*max(1,abs(values).max()):raise ValueError('KDE needs nonconstant data')
                    xx=np.linspace(values.min(),values.max(),200);ax.plot(xx,gaussian_kde(values)(xx),label=label)
                    ax.set(xlabel='Enstrophy',ylabel='Uniform-time sample density')
                else:
                    if threshold is None:raise ValueError('Specify an explicit event threshold')
                    row['upcrossings']=int(np.sum((values[:-1]<=threshold)&(values[1:]>threshold)))
                    ax.plot(grid,values,label=label);ax.axhline(threshold,color='grey',linestyle='--');ax.set(xlabel='Physical time',ylabel='Enstrophy')
                row.update(uniform_mean=float(np.mean(values)),uniform_samples=len(values))
            rows.append(row)
        for a,y in zip(axes,['Enstrophy: 0.5 integral omega^2','Energy: 0.5 integral |u|^2']):
            a.set(xlabel='Physical time',ylabel=y);a.legend(fontsize=7);a.grid(alpha=.2)
        fig.savefig(directory/'figures/diagnostics.png',dpi=150);plt.close(fig)
        if detail:
            ax.legend(fontsize=7);detail.savefig(directory/f'figures/{recipe}.png',dpi=150);plt.close(detail)
        with (directory/'tables/summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (directory/'analysis.log').write_text('COMPLETED: loaded explicit saved inputs; no integration performed\n')
        record.update(status='completed',outputs=[str(p.relative_to(directory)) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='analysis.json'])
        (directory/'analysis.log').write_text('COMPLETED: loaded explicit saved inputs; no integration performed\n')
        save();return directory
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}');save()
        (directory/'analysis.log').write_text('FAILED: '+str(exc)+'\n');plt.close('all');raise


def main(experiment,argv=None):
    p=argparse.ArgumentParser(description='Analyze explicit saved runs; no computation fallback')
    p.add_argument('--runs',nargs='+',type=Path);p.add_argument('--batch',type=Path)
    p.add_argument('--reference',type=Path);p.add_argument('--recipe',default='diagnostics',choices=['diagnostics','errors','spectrum','pchip_kde','events'])
    p.add_argument('--output-root',type=Path);p.add_argument('--window',nargs=2,type=float)
    p.add_argument('--sample-dt',type=float);p.add_argument('--threshold',type=float)
    p.add_argument('--allow-failed',action='store_true',help='Read available finite accepted prefixes; mark failed runs explicitly')
    a=p.parse_args(argv)
    if bool(a.runs)==bool(a.batch):p.error('Select exactly one of --runs or --batch')
    paths=a.runs
    if a.batch:
        members=json.loads((a.batch/'manifest.json').read_text())['members']
        if any(m['status']!='completed' for m in members):raise ValueError('Batch contains incomplete/failed members')
        paths=[m['path'] for m in members]
    print(analyze(paths,output_root=a.output_root,name=experiment,recipe=a.recipe,reference=a.reference,
                  window=a.window,sample_dt=a.sample_dt,threshold=a.threshold,allow_failed=a.allow_failed));return 0
