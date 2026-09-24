"""Frozen numerical baselines plus driver contracts. These are not full PDE convergence studies."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
BASELINE = Path(__file__).with_name("fixtures") / "solver_baseline.npz"
from solver import (State, Trial, History, Scheme, Assessment, AdaptiveAlgorithm,
                            FourierNS, Result, IntegrationError, integrate)
from solver.schemes import SDIRK2, SDIRK2MrSAV, ETDMS2, ETDMrSAVMS2B, ETDRK4, ETDMrSAVMS2L, MrSAVBDF2, IMEXEuler, LegacyLinearETD
from solver.adaptivity import EmbeddedErrorControl, SAVControl, StepDoubling, ProportionalController


class NumericalRegressionTests(unittest.TestCase):
    def setup_case(self, method, amplitude=.1):
        n, nu, gamma = 32, .025, 1000.
        force = lambda x, y, t: np.cos(y)*(1+.1*np.sin(t))
        model = FourierNS(nu, (n,n), forcing=force)
        x, y = model.X[:-1,:-1], model.Y[:-1,:-1]
        omega = amplitude*(np.cos(x)*np.cos(2*y)+.3*np.sin(3*x+y))
        initial = model.initial_state(omega)
        return model, initial, None

    def test_fixed_and_prescribed_parity_all_migrated_formats(self):
        pairs = [(SDIRK2(),"IMEX_RK2"),(SDIRK2MrSAV(),"SDIRK2_mr_SAV"),
                 (ETDMS2(),"ETDMS2"),(ETDMrSAVMS2B(),"ETD_mrSAV_MS2_b"),(ETDRK4(),"ETDRK4"),(ETDMrSAVMS2L(),"ETD_mrSAV_MS2_L"),
                 (MrSAVBDF2(),"mr_SAV_BDF2"),(IMEXEuler(),"IMEX"),(LegacyLinearETD(),"ETD")]
        for scheme, method in pairs:
            for mode in ("fixed","prescribed"):
                with self.subTest(scheme=scheme.name,mode=mode):
                    model, state, old = self.setup_case(method)
                    steps = np.full(8,.001) if mode=="fixed" else np.array([.001,.0007,.0012,.0011,.001,.0013,.0008,.0009])
                    times = np.r_[0,np.cumsum(steps)]
                    with np.load(BASELINE, allow_pickle=False) as saved:
                        old = {k: saved[f"{method}_{mode}_{k}"] for k in
                               ["Omega", "tn", "q", "Energy", "Enstrophy", "Energy_rate", "Palinstrophy"]}
                    args = {"dt":.001} if mode=="fixed" else {"steps":steps}
                    new = integrate(model,scheme,state,(0,.008 if mode=="fixed" else float(times[-1])),snapshots=times,**args)
                    np.testing.assert_allclose(new.snapshot_fields["omega"],old["Omega"],rtol=2e-12,atol=2e-13)
                    np.testing.assert_allclose(new.times,old["tn"],rtol=0,atol=2e-15)
                    if state.aux or scheme.auxiliary_defaults:
                        np.testing.assert_allclose(new.auxiliary["q"],old["q"],rtol=2e-12,atol=2e-13)
                    else:self.assertEqual(new.auxiliary,{})
                    for new_name,old_name in [("energy","Energy"),("enstrophy","Enstrophy"),("energy_rate","Energy_rate"),("palinstrophy","Palinstrophy")]:
                        np.testing.assert_allclose(new.diagnostics[new_name],old[old_name],rtol=2e-12,atol=2e-13)

    def test_adaptive_legacy_policy_nodes_and_counters(self):
        for scheme,method in [(SDIRK2MrSAV(),"SDIRK2_mr_SAV"),(ETDMrSAVMS2B(),"ETD_mrSAV_MS2_b")]:
            with self.subTest(method=method):
                model,state,old = self.setup_case(method,amplitude=2)
                with np.load(BASELINE, allow_pickle=False) as saved:
                    old = {k: saved[f"{method}_adaptive_{k}"] for k in
                           ["Omega", "tn", "q", "accepted_steps", "rejected_steps", "forced_accept_steps"]}
                control = SAVControl({"omega":(1e-12,1e-5)},{"q":(1.,1e-3)})
                new = integrate(model,scheme,state,(0,.02),adaptive=control,initial_dt=.0005,min_dt=1e-5,max_dt=.002)
                np.testing.assert_allclose(new.times,old["tn"],rtol=1e-9,atol=1e-11)
                np.testing.assert_allclose(new.final_state.fields["omega"],old["Omega"],rtol=2e-10,atol=2e-11)
                np.testing.assert_allclose(new.auxiliary["q"],old["q"],rtol=2e-10,atol=2e-11)
                for key in ("accepted_steps","rejected_steps","forced_accept_steps"):
                    self.assertEqual(new.stats[key],old[key])

    def test_forced_fourier_mode_exact_solution_and_temporal_refinement(self):
        model=FourierNS(.2,(16,16),forcing=lambda x,y,t:np.cos(x))
        state=model.initial_state(np.cos(model.X[:-1,:-1]))
        exact=(np.exp(-.2)+(1-np.exp(-.2))/.2)*state.fields["omega"]
        reference=integrate(model,ETDRK4(),state,(0,1),dt=.1)
        np.testing.assert_allclose(reference.final_state.fields["omega"],exact,rtol=0,atol=2e-12)
        errors=[]
        for dt in (.1,.05,.025):
            value=integrate(model,SDIRK2(),state,(0,1),dt=dt)
            errors.append(np.max(abs(value.final_state.fields["omega"]-exact)))
        self.assertGreater(errors[0]/errors[1],3.8)
        self.assertGreater(errors[1]/errors[2],3.8)

    def test_step_doubling_explicit_option_for_plain_sdirk2(self):
        model,state,_=self.setup_case("IMEX_RK2")
        result=integrate(model,SDIRK2(),state,(0,.02),
                         adaptive=StepDoubling({"omega":(1e-12,1e-6)},ProportionalController(exponent=1/3)),initial_dt=.002)
        self.assertEqual(result.stats["status"],"completed")
        self.assertEqual(result.auxiliary,{})
        with self.assertRaises(ValueError):
            integrate(model,ETDMS2(),state,(0,.02),adaptive=StepDoubling({"omega":(1e-12,1e-6)}),initial_dt=.002)


class ToyModel:
    def norm(self,name,value):return float(np.linalg.norm(value))


class Translation(Scheme):
    name="Translation"
    variable_step=True
    order=1
    def step(self,model,history,dt):
        return Trial(State({k:a+dt for k,a in history.state.fields.items()},history.state.aux))


class Scheduled(AdaptiveAlgorithm):
    def __init__(self,accept=True,next_dt=.2):
        self.accept,self.next_dt=accept,next_dt
        self.attempted=[]
        self.accepted=[]
        self.rejected=[]
    def attempt(self,model,scheme,history,dt):
        self.attempted.append((history,dt))
        return Assessment(scheme.step(model,history,dt),self.accept,self.next_dt,{"error":0. if self.accept else 2.})
    def on_accept(self,value,forced):self.accepted.append(forced)
    def on_reject(self,value):self.rejected.append(value)


class DriverContractTests(unittest.TestCase):
    def setUp(self):
        self.state=State({"a":np.zeros(3),"b":np.zeros((2,2))})
        self.model=ToyModel()

    def test_nearest_nodes_tie_earlier_duplicates_and_unsorted_requests(self):
        requests=[.9,.1,.51,.49,.5,0,1,.5]
        result=integrate(self.model,Translation(),self.state,(0,1),adaptive=Scheduled(next_dt=.2),initial_dt=.2,snapshots=requests)
        np.testing.assert_allclose(result.actual_snapshot_times,[.8,0,.6,.4,.4,0,1,.4],rtol=0,atol=1e-14)
        self.assertEqual(result.snapshot_indices[4],result.snapshot_indices[7])
        for index,time in enumerate(result.snapshot_times):
            np.testing.assert_allclose(result.snapshot_fields["a"][index],time,atol=1e-14)
        self.assertEqual(len(result.snapshot_times),5)

    def test_fixed_prescribed_strict_grid_and_endpoint(self):
        for settings in ({"dt":.2},{"steps":[.2]*5}):
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError,"actual grid"):
                    integrate(self.model,Translation(),self.state,(0,1),snapshots=[.1],**settings)
                good=integrate(self.model,Translation(),self.state,(0,1),snapshots=[.4,1],**settings)
                np.testing.assert_allclose(good.actual_snapshot_times,[.4,1],atol=1e-14)
        with self.assertRaises(ValueError):integrate(self.model,Translation(),self.state,(0,1),dt=.3)
        with self.assertRaises(ValueError):integrate(self.model,Translation(),self.state,(0,1),steps=[.2,.3])

    def test_snapshot_selection_at_very_small_physical_times(self):
        value=integrate(self.model,Translation(),self.state,(0,1e-15),
                        adaptive=Scheduled(next_dt=2e-16),initial_dt=2e-16,snapshots=[3e-16,1e-15])
        np.testing.assert_allclose(value.actual_snapshot_times,[2e-16,1e-15],rtol=1e-14,atol=0)
        with self.assertRaises(ValueError):
            integrate(self.model,Translation(),self.state,(0,1e-15),dt=7e-16)

    def test_lower_bound_force_and_upper_bound_not_force(self):
        algorithm=Scheduled(accept=False,next_dt=.01)
        result=integrate(self.model,Translation(),self.state,(0,.25),adaptive=algorithm,initial_dt=.2,min_dt=.1,max_dt=.2)
        self.assertEqual(result.stats["rejected_steps"],1)
        self.assertEqual(result.stats["forced_accept_steps"],3)
        np.testing.assert_allclose(result.steps,[.1,.1,.05])
        self.assertEqual(algorithm.accepted,[True,True,True])

    def test_no_bounds_rejection_preserves_history_and_controller_hooks(self):
        class Retry(Scheduled):
            def attempt(self,model,scheme,history,dt):
                self.accept=dt<=.025
                self.next_dt=dt if self.accept else dt/2
                return super().attempt(model,scheme,history,dt)
        algorithm=Retry()
        result=integrate(self.model,Translation(),self.state,(0,.1),adaptive=algorithm,initial_dt=.1)
        self.assertEqual(result.stats["rejected_steps"],2)
        self.assertEqual(result.stats["forced_accept_steps"],0)
        self.assertIs(algorithm.attempted[0][0],algorithm.attempted[1][0])
        self.assertIs(algorithm.attempted[1][0],algorithm.attempted[2][0])
        np.testing.assert_allclose(result.final_state.fields["a"],.1)
        with self.assertRaises(ValueError):algorithm.attempted[0][0].state.fields["a"][0]=4

    def test_upper_bound_alone_and_bounds_absent(self):
        algorithm=Scheduled(next_dt=1)
        capped=integrate(self.model,Translation(),self.state,(0,1),adaptive=algorithm,initial_dt=1,max_dt=.2)
        self.assertLessEqual(max(capped.steps),.2)
        free=integrate(self.model,Translation(),self.state,(0,1),adaptive=Scheduled(next_dt=1),initial_dt=.3)
        np.testing.assert_allclose(free.steps,[.3,.7])

    def test_multiple_scalar_auxiliaries_and_custom_startup(self):
        class TwoStep(Translation):
            history_size=2
            auxiliary_defaults={"q":1.,"r":2.}
            def startup(self,model,history,dt):raise AssertionError("override not used")
        starts=[]
        def starter(model,history,dt):
            starts.append(dt)
            return Trial(State({k:v+2*dt for k,v in history.state.fields.items()},history.state.aux))
        result=integrate(self.model,TwoStep(),self.state,(0,.3),steps=[.1]*3,startup=starter)
        self.assertEqual(starts,[.1])
        self.assertEqual(set(result.auxiliary),{"q","r"})
        np.testing.assert_allclose(result.final_state.fields["b"],.4)
        with self.assertRaises(ValueError):State({"a":np.ones(2)},{"q":np.ones(2)})

    def test_trial_failure_partial_save_reload_and_no_overwrite(self):
        class Bad(Translation):
            def step(self,model,history,dt):
                if history.time>=.1:return Trial(State({k:np.full_like(v,np.nan) for k,v in history.state.fields.items()}))
                return super().step(model,history,dt)
        with self.assertRaises(IntegrationError) as caught:
            integrate(self.model,Bad(),self.state,(0,.3),dt=.1,snapshots=[0,.3])
        partial=caught.exception.result
        self.assertEqual(partial.stats["status"],"failed")
        self.assertAlmostEqual(partial.times[-1],.1)
        self.assertFalse(partial.failed_state.finite())
        self.assertEqual(partial.snapshot_indices[-1],-1)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"failure.npz"
            partial.save(path,{"test":"failure"})
            loaded=Result.load(path)
            np.testing.assert_array_equal(loaded.times,partial.times)
            self.assertFalse(loaded.failed_state.finite())
            self.assertEqual(loaded.metadata["test"],"failure")
            with self.assertRaises(FileExistsError):partial.save(path)

    def test_missing_estimator_and_no_time_progress_fail_explicitly(self):
        with self.assertRaises(ValueError):
            integrate(self.model,Translation(),self.state,(0,1),adaptive=EmbeddedErrorControl({"a":(1e-6,1e-3)}),initial_dt=.1)
        with self.assertRaises(IntegrationError):
            integrate(self.model,Translation(),self.state,(1,2),adaptive=Scheduled(),initial_dt=1e-20)

    def test_diagnostic_failure_keeps_aligned_accepted_prefix(self):
        def diagnostic(state,time):
            if time>=.2:raise RuntimeError("diagnostic failed")
            return {"value":time}
        with self.assertRaises(IntegrationError) as caught:
            integrate(self.model,Translation(),self.state,(0,.3),dt=.1,diagnostics=diagnostic)
        result=caught.exception.result
        self.assertEqual(len(result.times),len(result.diagnostics["value"]))
        self.assertTrue(np.isnan(result.diagnostics["value"][-1]))

    def test_completed_multifield_scalar_result_roundtrip(self):
        class Scalars(Translation):auxiliary_defaults={"q":1.,"r":2.}
        result=integrate(self.model,Scalars(),self.state,(0,1),dt=.2,snapshots=[0,.4,1])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"result.npz"
            result.save(path)
            loaded=Result.load(path)
        np.testing.assert_array_equal(loaded.times,result.times)
        np.testing.assert_array_equal(loaded.snapshot_indices,result.snapshot_indices)
        for name in self.state.fields:
            np.testing.assert_array_equal(loaded.snapshot_fields[name],result.snapshot_fields[name])
            np.testing.assert_array_equal(loaded.final_state.fields[name],result.final_state.fields[name])
        self.assertEqual(dict(loaded.final_state.aux),{"q":1.,"r":2.})


if __name__=="__main__":unittest.main()
