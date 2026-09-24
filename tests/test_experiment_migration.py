"""Short end-to-end storage/analysis checks for the migrated framework."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.common import workflow as w
from experiments.common.analysis import analyze,load_run
import h5py
import numpy as np


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_reuse_and_analysis_never_compute_or_change_inputs(self):
        c=w.config('bursting');p=w.run('bursting',c,self.root/'runs')
        hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file()}
        with patch.object(w,'compute',side_effect=AssertionError('Unexpected integration')):
            self.assertEqual(w.run('bursting',c,self.root/'runs'),p)
            a=analyze([p],output_root=self.root/'reports')
            b=analyze([p],output_root=self.root/'reports')
        self.assertNotEqual(a,b)
        self.assertEqual(json.loads((a/'analysis.json').read_text())['status'],'completed')
        self.assertEqual(hashes,{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file()})
        self.assertNotEqual(w.run('bursting',c,self.root/'runs',True),p)

    def test_failed_batch_can_retry_only_failed_member(self):
        c=w.config('bursting');real=w.compute
        def compute(c,log):
            if c['scheme']=='sdirk2':raise RuntimeError('Injected failure')
            return real(c,log)
        with patch.object(w,'compute',side_effect=compute):
            b=w.batch('bursting',c,[{'id':'ok','parameters':{}},{'id':'retry','parameters':{'scheme':'sdirk2'}}],self.root/'runs')
        original=json.loads((b/'manifest.json').read_text());self.assertEqual(original['status'],'failed')
        failed_path=Path(original['members'][1]['attempts'][0])
        self.assertEqual(json.loads((failed_path/'manifest.json').read_text())['status'],'failed')
        w.retry_batch(b,['retry'])
        now=json.loads((b/'manifest.json').read_text());self.assertEqual(now['status'],'completed')
        self.assertEqual(now['members'][0],original['members'][0]);self.assertEqual(len(now['members'][1]['attempts']),2)
        with self.assertRaises(ValueError):w.retry_batch(b,['ok'])

    def test_failed_analysis_and_identity_validation(self):
        p=w.run('convergence',w.config('convergence'),self.root/'runs')
        m=json.loads((p/'manifest.json').read_text());m['identity']['parameters']['gamma']=12
        w.write_json(p/'manifest.json',m)
        with self.assertRaises(ValueError):analyze([p],output_root=self.root/'reports')
        record=next((self.root/'reports').rglob('analysis.json'))
        self.assertEqual(json.loads(record.read_text())['status'],'failed')

    def test_legacy_hdf5_and_reference_errors(self):
        p=self.root/'legacy-fixture';p.mkdir()
        identity={'schema':1,'fixture':'HDF5 loader'}
        w.write_json(p/'config.json',{'parameters':{'N':4,'T':.004}})
        w.write_json(p/'manifest.json',{'run_id':p.name,'status':'completed','identity':identity})
        with h5py.File(p/'results.h5','w') as f:
            f.attrs.update(status='completed',run_id=p.name,identity=json.dumps(identity))
            f['tn']=np.linspace(0,.004,5);f['Energy']=np.ones(5);f['Enstrophy']=np.ones(5)
            f['Omega']=np.zeros((1,4,4))
        loaded=load_run(p);self.assertEqual(len(loaded['time']),5)
        analyze([p],output_root=self.root/'reports')
        c=w.config('convergence');r=w.run('convergence',w.merge(c,{'scheme':'etdrk4','dt':.00025}),self.root/'runs')
        p=w.run('convergence',c,self.root/'runs')
        report=analyze([p],reference=r,recipe='errors',output_root=self.root/'reports')
        self.assertIn('omega_l2_error',(report/'tables/summary.csv').read_text())
        bad=w.run('convergence',w.merge(c,{'nu':.02}),self.root/'runs')
        with self.assertRaises(ValueError):analyze([bad],reference=r,recipe='errors',output_root=self.root/'reports')

    def test_archive_preserved_byte_for_byte(self):
        for entry in json.loads((w.ROOT/'archive/legacy/migration.json').read_text()):
            self.assertEqual(hashlib.sha256((w.ROOT/entry['archived']).read_bytes()).hexdigest(),entry['sha256'])

if __name__=='__main__':unittest.main()

class FailedPrefixTests(unittest.TestCase):
    def test_explicit_failure_analysis_has_no_fabricated_endpoint_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=w.merge(w.config('convergence'),{'max_steps':2})
            with self.assertRaises(Exception) as raised:w.run('convergence',c,root/'runs')
            path=raised.exception.run_directory
            with self.assertRaises(ValueError):load_run(path)
            loaded=load_run(path,allow_failed=True)
            self.assertLess(loaded['time'][-1],c['T'])
            report=analyze([path],allow_failed=True,output_root=root/'reports')
            self.assertIn('failed',(report/'tables/summary.csv').read_text())
