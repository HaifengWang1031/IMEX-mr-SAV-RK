"""Small, temporary-directory checks of extraction and persistence, not PDE convergence."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import numpy as np
from run import Case, METHODS, make_solver, run


def main():
    c=Case(grid=32,modes=6,final_time=.04)
    # One Fourier mode has zero advection; forced linear evolution is explicit.
    linear=make_solver(replace(c,nu=.2),'ETDRK4')
    linear.Omega0=np.cos(linear.X[:-1,:-1])
    with contextlib.redirect_stdout(io.StringIO()):
        linear.solve_fix_step((0,1.),.1)
    exact=(np.exp(-.2)+(1-np.exp(-.2))/.2)*np.cos(linear.X[:-1,:-1])
    np.testing.assert_allclose(linear.Omega[-1],exact,rtol=0,atol=1e-12)
    print('ETDRK4 forced single-mode exact solution passed')
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        for method in (*METHODS,'ETDRK4'):
            result=run(c,method,8,folder)
            solver=make_solver(c,method)
            with contextlib.redirect_stdout(io.StringIO()):
                solver.solve_fix_step((0,c.final_time),c.final_time/8,
                                      snapshot=np.linspace(0,c.final_time,5))
            np.testing.assert_allclose(result['omega'],solver.Omega,rtol=1e-13,atol=1e-14)
            np.testing.assert_array_equal(result['omega'],run(c,method,8,folder)['omega'])
            print(method,'driver parity / save / reuse passed')
        endpoint=run(c,'IMEX_RK2',7,folder)
        np.testing.assert_array_equal(endpoint['times'],[0,c.final_time])
        meta=json.loads(str(endpoint['metadata']))
        assert meta['config']['tau']==c.final_time/7
        path=folder/(meta['run_id']+'.npz')
        bad=dict(endpoint)
        bad_meta=json.loads(str(bad['metadata']));bad_meta['config']['case']['nu']=.2
        bad['metadata']=np.array(json.dumps(bad_meta))
        np.savez(path,**bad)
        try:
            run(c,'IMEX_RK2',7,folder)
        except ValueError:
            print('metadata mismatch rejected')
        else:
            raise AssertionError('metadata mismatch was not rejected')
        # A manifest without a completed NPZ restarts from the initial condition.
        path.unlink()
        rerun=run(c,'IMEX_RK2',7,folder)
        np.testing.assert_array_equal(rerun['omega'],endpoint['omega'])
        badcase=replace(c,grid=64,modes=10,amplitude=4,final_time=2)
        failed=run(badcase,'IMEX_RK2',16,folder)
        fm=json.loads(str(failed['metadata']))
        assert fm['status']=='solution_blowup'
        assert fm['last_state_time']<badcase.final_time
        assert np.isfinite(failed['diagnostics']).all()
        np.testing.assert_array_equal(run(badcase,'IMEX_RK2',16,folder)['diagnostics'],failed['diagnostics'])
        print('failure prefix and terminal status persisted; failed run reused explicitly')
    print('Workflow checks passed. These do not establish PDE accuracy or spatial convergence.')


if __name__=='__main__':
    main()
