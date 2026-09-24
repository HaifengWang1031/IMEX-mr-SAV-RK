"""Run/batch persistence for explicit, configurable nextgen experiments."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import time
import uuid
import numpy as np
from solver import integrate, IntegrationError
from solver.schemes import SDIRK2, SDIRK2MrSAV, ETDRK4, ETDMS2, ETDMrSAVMS2B, ETDMrSAVMS2L, IMEXEuler, LegacyLinearETD, MrSAVBDF2
from solver.adaptivity import EmbeddedErrorControl, SAVControl, StepDoubling, ProportionalController
from .model import build

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ('bursting','convergence','mean_reverting','adaptive_tolerance','shearflow','spatial_spectrum','sdirk2_transition')


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def merge(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        if key not in result:
            raise ValueError(f'Unknown configuration key: {key}')
        result[key] = merge(result[key], value) if isinstance(value, dict) else value
    return result


def config(experiment, path=None):
    if experiment not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment: {experiment}')
    value = json.loads((ROOT/'experiments'/experiment/'configs/default.json').read_text())
    if path:
        value = merge(value, json.loads(Path(path).read_text()))
    canonical(value)
    return value


def scheme(c):
    choices = {'imex_euler':IMEXEuler, 'legacy_linear_etd':LegacyLinearETD, 'mrsav_bdf2':lambda:MrSAVBDF2(c['gamma']), 'sdirk2': SDIRK2, 'sdirk2_mrsav': lambda: SDIRK2MrSAV(c['gamma']),
               'etdrk4': ETDRK4, 'etdms2': ETDMS2,
               'etd_mrsav_ms2_b': lambda: ETDMrSAVMS2B(c['gamma']),
               'etd_mrsav_ms2_l': lambda: ETDMrSAVMS2L(c['gamma'], c['talbot_nodes'])}
    if c['scheme'] not in choices:
        raise ValueError(f"Unsupported scheme {c['scheme']}; no implicit method substitution")
    if c['root_selection'] != 'legacy':
        raise ValueError('Only the legacy root selection is implemented in these schemes')
    return choices[c['scheme']]()


def compute(c, logger):
    model, initial = build(c)
    method = scheme(c)
    def progress(e):
        logger.info('t=%g/%g accepted=%d rejected=%d status=%s',e['time'],e['end'],e['accepted_steps'],e['rejected_steps'],e['status'])
    if c['warmup_time']:
        end = np.ceil(c['warmup_time']/c['warmup_dt'])*c['warmup_dt']
        initial = integrate(model, ETDRK4(), initial, (0,end), dt=c['warmup_dt'], progress=progress).final_state
    if c['mode'] == 'fixed':
        options = {'dt': c['dt']}
    elif c['mode'] == 'prescribed':
        options = {'steps': c['steps']}
    elif c['mode'] == 'adaptive':
        a = c['adaptive']; tol = {'omega': (a['atol'], a['rtol'])}
        controller = ProportionalController(a['exponent'], a['safety'], a['max_growth'])
        choices = {'embedded': lambda: EmbeddedErrorControl(tol, controller),
                   'sav': lambda: SAVControl(tol, {'q':(1,a['q_tolerance'])}, a['exponent'], a['safety'], a['max_growth']),
                   'doubling': lambda: StepDoubling(tol, controller)}
        if a['algorithm'] not in choices: raise ValueError('Unknown adaptive algorithm')
        options = {'adaptive': choices[a['algorithm']](), 'initial_dt':a['initial_dt'],
                   'min_dt':a['min_dt'], 'max_dt':a['max_dt']}
    else:
        raise ValueError('Unknown time-step mode')
    return integrate(model, method, initial, (0,c['T']), snapshots=c['snapshots'],
                     progress=progress, log_interval=c['log_interval'], max_steps=c['max_steps'], **options)


def sources():
    paths = list((ROOT/'solver').rglob('*.py')) + list((ROOT/'experiments/common').glob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run(experiment, c, output_root=None, rerun=False):
    import scipy, pyfftw
    identity = {'schema':2, 'experiment':experiment,
                'parameters':{k:v for k,v in c.items() if k not in ('log_interval','description')},
                'source_sha256':sources(),
                'environment':{'python':platform.python_version(),'numpy':np.__version__,
                               'scipy':scipy.__version__,'pyfftw':pyfftw.__version__}}
    key = hashlib.sha256(canonical(identity).encode()).hexdigest()[:16]
    output = Path(output_root or ROOT/'runs')/experiment
    output.mkdir(parents=True, exist_ok=True)
    if not rerun:
        for path in sorted(output.glob(key+'-*/manifest.json')):
            old = json.loads(path.read_text())
            if old['status'] == 'completed' and old['identity'] == identity:
                from .analysis import load_run
                load_run(path.parent)  # Validate saved identity and required numerical output.
                return path.parent
    run_id = key+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    directory = output/run_id; directory.mkdir()
    write_json(directory/'config.json',c)
    manifest = {'run_id':run_id,'identity':identity,'status':'running','result':'results.npz'}
    write_json(directory/'manifest.json',manifest)
    logger = logging.getLogger(run_id);logger.setLevel(logging.INFO)
    handler = logging.FileHandler(directory/'run.log');handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler);start=time.perf_counter()
    try:
        logger.info('STARTED %s %s',experiment,canonical(c))
        result = compute(c,logger)
        result.save(directory/'results.npz',{'identity':identity,'run_id':run_id})
        manifest.update(status='completed',stats=result.stats)
        logger.info('COMPLETED %s',directory)
    except Exception as exc:
        if isinstance(exc,IntegrationError):
            try:
                exc.result.save(directory/'results.npz',{'identity':identity,'run_id':run_id})
            except Exception:
                logger.exception('Failed to save accepted prefix')
        exc.run_directory = directory
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        logger.exception('FAILED')
        raise
    finally:
        manifest['elapsed_wall_seconds']=time.perf_counter()-start
        write_json(directory/'manifest.json',manifest)
        logger.removeHandler(handler);handler.close()
    return directory


def batch(experiment, base, cases, output_root=None, rerun=False):
    ids=[case['id'] for case in cases]
    if not ids or len(set(ids))!=len(ids): raise ValueError('Batch case ids must be nonempty and unique')
    configs=[merge(base,case['parameters']) for case in cases]
    root=Path(output_root or ROOT/'runs')/experiment/'batches'
    directory=root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]);directory.mkdir(parents=True)
    write_json(directory/'config.json',{'base':base,'cases':cases})
    record={'status':'running','members':[{'case_id':i,'status':'pending','attempts':[]} for i in ids]}
    write_json(directory/'manifest.json',record)
    for member,c in zip(record['members'],configs):
        member['status']='running';write_json(directory/'manifest.json',record)
        try:
            began=time.time_ns()
            path=run(experiment,c,output_root,rerun)
            member.update(status='completed',run_id=path.name,path=str(path.resolve()),reused=path.stat().st_mtime_ns<began)
            member['attempts'].append(str(path.resolve()))
        except Exception as exc:
            member.update(status='failed',error=str(exc))
            if hasattr(exc,'run_directory'):
                member['attempts'].append(str(exc.run_directory.resolve()))
        with (directory/'batch.log').open('a') as f:f.write(canonical(member)+'\n')
        write_json(directory/'manifest.json',record)
    record['status']='completed' if all(m['status']=='completed' for m in record['members']) else 'failed'
    write_json(directory/'manifest.json',record)
    return directory


def retry_batch(directory, case_ids):
    directory=Path(directory).resolve()
    record=json.loads((directory/'manifest.json').read_text())
    saved=json.loads((directory/'config.json').read_text())
    selected=set(case_ids)
    members={m['case_id']:m for m in record['members']}
    if not selected or not selected <= members.keys():raise ValueError('Unknown retry case ids')
    if any(members[i]['status']=='completed' for i in selected):raise ValueError('Retry selects failed/incomplete members only')
    experiment=directory.parent.parent.name
    output_root=directory.parent.parent.parent
    for case in saved['cases']:
        if case['id'] not in selected:continue
        member=members[case['id']];member['status']='running';record['status']='running'
        write_json(directory/'manifest.json',record)
        try:
            path=run(experiment,merge(saved['base'],case['parameters']),output_root,rerun=True)
            member['attempts'].append(str(path.resolve()))
            member.update(status='completed',run_id=path.name,path=str(path.resolve()),reused=False)
            member.pop('error',None)
        except Exception as exc:
            member.update(status='failed',error=str(exc))
            if hasattr(exc,'run_directory'):member['attempts'].append(str(exc.run_directory.resolve()))
        with (directory/'batch.log').open('a') as f:f.write('RETRY '+canonical(member)+'\n')
        write_json(directory/'manifest.json',record)
    record['status']='completed' if all(m['status']=='completed' for m in record['members']) else 'failed'
    write_json(directory/'manifest.json',record)
    return directory


def main(experiment, argv=None):
    parser=argparse.ArgumentParser(description=f'{experiment}: explicit nextgen computation; defaults are smoke-sized')
    parser.add_argument('--config',type=Path);parser.add_argument('--batch',type=Path)
    parser.add_argument('--output-root',type=Path);parser.add_argument('--rerun',action='store_true')
    parser.add_argument('--show-config',action='store_true')
    parser.add_argument('--retry-batch',type=Path);parser.add_argument('--cases',nargs='+')
    args=parser.parse_args(argv);c=config(experiment,args.config)
    if args.retry_batch:
        if not args.cases:parser.error('--retry-batch requires --cases')
        path=retry_batch(args.retry_batch,args.cases);print(path)
        return 0 if json.loads((path/'manifest.json').read_text())['status']=='completed' else 1
    if args.show_config:print(json.dumps(c,indent=2));return 0
    if args.batch:
        path=batch(experiment,c,json.loads(args.batch.read_text())['cases'],args.output_root,args.rerun)
        print(path)
        return 0 if json.loads((path/'manifest.json').read_text())['status']=='completed' else 1
    print(run(experiment,c,args.output_root,args.rerun));return 0
