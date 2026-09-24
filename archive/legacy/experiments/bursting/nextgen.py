"""Pilot migration of the bursting experiment to the independent nextgen interface.

Controls: same Fourier streamfunction perturbation, m*cos(m*y) forcing and ETDRK4
warmup as the established bursting experiment. Select ordinary/SAV schemes without
changing the initial physical field. Observables are physical-time diagnostics,
scalar auxiliaries, accepted-node snapshots and adaptive counts. A short successful
run validates the workflow, not long-time bursting statistics or PDE convergence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import uuid
import warnings
import numpy as np
import pyfftw
import scipy

from .run import ROOT, atomic_json, canonical, initial_streamfunction
from solver.nextgen import FourierNS, integrate, IntegrationError
from solver.nextgen.schemes import SDIRK2, SDIRK2MrSAV, ETDMS2, ETDMrSAVMS2B, ETDRK4
from solver.nextgen.adaptivity import EmbeddedErrorControl, SAVControl, StepDoubling, ProportionalController

CONFIG_PATH = Path(__file__).with_name("nextgen_config.json")


def compute(c, logger):
    if c["Re"] <= 0 or c["m"] <= 0 or c["warmup_time"] < 0 or c["warmup_dt"] <= 0:
        raise ValueError("Invalid model/warmup settings")
    force = lambda x, y, t: c["m"]*np.cos(c["m"]*y)
    model = FourierNS(1/c["Re"], (c["N"],c["N"]), forcing=force, threads=c["threads"])
    x,y = model.X[:-1,:-1],model.Y[:-1,:-1]
    psi = initial_streamfunction(x,y,model.nu,c["m"],c["eps"])
    u,v = model.stream2velocity(psi)
    initial = model.initial_state(model.velocity2vorticity(u,v))
    def progress(event):
        logger.info("t=%.10g/%g accepted=%d rejected=%d forced=%d elapsed_wall=%.2fs status=%s",
                    event["time"],event["end"],event["accepted_steps"],event["rejected_steps"],
                    event["forced_accept_steps"],event["elapsed_wall"],event["status"])
    warmup_end = float(np.ceil(c["warmup_time"]/c["warmup_dt"])*c["warmup_dt"])
    if c["warmup_time"]:
        logger.info("Warmup: requested=%g actual=%g",c["warmup_time"],warmup_end)
        initial = integrate(model,ETDRK4(),initial,(0,warmup_end),dt=c["warmup_dt"],
                            progress=progress,log_interval=c["log_interval"]).final_state
    # Experiment composition is explicit; integrate() contains no method-name dispatch.
    schemes = {"sdirk2":SDIRK2, "sdirk2_mrsav":lambda:SDIRK2MrSAV(c["gamma"]),
               "etdms2":ETDMS2, "etd_mrsav_ms2_b":lambda:ETDMrSAVMS2B(c["gamma"])}
    scheme = schemes[c["scheme"]]()
    if c["mode"]=="fixed":
        options={"dt":c["dt"]}
    elif c["mode"]=="prescribed":
        options={"steps":c["steps"]}
    elif c["mode"]=="adaptive":
        tolerances={"omega":(c["atol"],c["rtol"])}
        controller=ProportionalController(c["control_exponent"],c["safety"],c["max_growth"])
        algorithms={"embedded":lambda:EmbeddedErrorControl(tolerances,controller),
                    "sav":lambda:SAVControl(tolerances,{"q":(1,c["rtol_q"])},c["control_exponent"],c["safety"],c["max_growth"]),
                    "doubling":lambda:StepDoubling(tolerances,controller)}
        options={"adaptive":algorithms[c["algorithm"]](),"initial_dt":c["initial_dt"],
                 "min_dt":c["min_dt"],"max_dt":c["max_dt"]}
    else:
        raise ValueError("Unknown mode")
    logger.info("Main integration: scheme=%s mode=%s",scheme.name,c["mode"])
    result=integrate(model,scheme,initial,(0,c["T"]),snapshots=c["snapshots"],
                     progress=progress,log_interval=c["log_interval"],**options)
    result.metadata["warmup_actual_end"]=warmup_end
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=CONFIG_PATH)
    parser.add_argument("--scheme",choices=["sdirk2","sdirk2_mrsav","etdms2","etd_mrsav_ms2_b"])
    parser.add_argument("--mode",choices=["fixed","prescribed","adaptive"])
    parser.add_argument("--algorithm",choices=["embedded","sav","doubling"])
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--rerun",action="store_true")
    parser.add_argument("--show-config",action="store_true")
    args=parser.parse_args(argv)
    c=json.loads(CONFIG_PATH.read_text())
    c.pop("_help",None)
    overrides=json.loads(args.config.read_text())
    if overrides.keys()-c.keys()-{"_help"}:
        parser.error("Unknown configuration keys")
    c.update({k:v for k,v in overrides.items() if k!="_help"})
    for key in ("scheme","mode","algorithm","output_root"):
        if getattr(args,key) is not None:c[key]=str(getattr(args,key))
    canonical(c)  # reject NaN/Infinity settings before creating a run
    if args.show_config:
        print(json.dumps(c,indent=2));return 0
    output=Path(c["output_root"]).expanduser()
    if not output.is_absolute():output=ROOT/output
    source_files=list((ROOT/"solver/nextgen").rglob("*.py"))+[Path(__file__),Path(__file__).with_name("run.py")]
    identity={"schema":1,"parameters":{k:v for k,v in c.items() if k not in ("output_root","log_interval")},
              "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
                             "pyfftw":pyfftw.__version__,"platform":platform.platform()}}
    signature=hashlib.sha256(canonical(identity).encode()).hexdigest()
    if not args.rerun:
        for manifest_path in sorted(output.glob(f"{signature[:16]}-*/manifest.json"),reverse=True):
            old=json.loads(manifest_path.read_text())
            if old["status"]!="completed" or old["identity"]!=identity:continue
            with np.load(manifest_path.with_name("results.npz"),allow_pickle=False) as data:
                info=json.loads(str(data["metadata"]))
                if info["stats"]["status"]!="completed" or info["metadata"]["identity"]!=identity:
                    raise ValueError("Completed record mismatch; inspect it or use --rerun")
            print(f"REUSED {manifest_path.parent}");return 0
    run_id=f"{signature[:16]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"
    directory=output/run_id
    directory.mkdir(parents=True)
    for name in ("figures","tables"):(directory/name).mkdir()
    atomic_json(directory/"config.json",c)
    manifest={"run_id":run_id,"identity":identity,"status":"running","config_file":str(args.config.resolve())}
    atomic_json(directory/"manifest.json",manifest)
    logger=logging.Logger(run_id,level=logging.INFO)
    handlers=[logging.FileHandler(directory/"run.log"),logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"));logger.addHandler(handler)
    try:
        logger.info("STARTED %s; effective_config=%s",directory,canonical(c))
        with warnings.catch_warnings():
            warnings.showwarning=lambda message,category,filename,lineno,file=None,line=None:logger.warning("%s: %s (%s:%d)",category.__name__,message,filename,lineno)
            result=compute(c,logger)
        result.save(directory/"results.npz",{"identity":identity,"run_id":run_id})
        manifest.update(status="completed",stats=result.stats)
        atomic_json(directory/"manifest.json",manifest)
        logger.info("COMPLETED results=%s",directory/"results.npz")
        return 0
    except Exception as exc:
        if isinstance(exc,IntegrationError):
            try:
                exc.result.save(directory/"results.npz",{"identity":identity,"run_id":run_id})
            except OSError:
                logger.exception("Could not persist the accepted prefix")
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        atomic_json(directory/"manifest.json",manifest)
        logger.exception("FAILED; record preserved at %s",directory)
        return 1
    finally:
        for handler in handlers:logger.removeHandler(handler);handler.close()


if __name__=="__main__":raise SystemExit(main())
