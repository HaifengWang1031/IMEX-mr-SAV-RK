"""Small real integrations and run-record failure paths; not PDE convergence tests."""
import contextlib
import copy
import hashlib
import io
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import h5py
import numpy as np

import legacy_bursting as bursting


class BurstingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bursting-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.argv = ["--N", "32", "--T", "0.004", "--tau", "0.001",
                     "--warmup-time", "0", "--snapshot-dt", "0.002",
                     "--output-root", str(self.root / "runs")]

    def config(self, *extra):
        return bursting.resolve_config(self.argv + list(extra))[0]

    def run_case(self, config=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return bursting.run_experiment(config or self.config(), **kwargs)

    def test_saved_effective_configuration_reuse_and_explicit_rerun(self):
        config = self.config("--gamma", "600")
        path = self.run_case(config)
        saved = json.loads((path / "config.json").read_text())
        self.assertEqual(saved["parameters"]["gamma"], 600)
        manifest = json.loads((path / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["steps"], 4)
        with h5py.File(path / "results.h5", "r") as result:
            np.testing.assert_allclose(result["tn_s"][:], [0, .002, .004], rtol=0, atol=1e-15)
            self.assertEqual(result["Omega"].shape, (3, 32, 32))
            self.assertEqual(result.attrs["gamma"], 600)
            self.assertEqual(result.attrs["run_id"], path.name)
            original = result["Omega"][:]
        log = (path / "run.log").read_bytes()
        data_hash = hashlib.sha256((path / "results.h5").read_bytes()).hexdigest()
        with patch.object(bursting, "integrate", side_effect=AssertionError("must not solve on reuse")):
            self.assertEqual(self.run_case(config), path)
        self.assertEqual((path / "run.log").read_bytes(), log)
        new = self.run_case(config, rerun=True)
        self.assertNotEqual(path, new)
        self.assertEqual(hashlib.sha256((path / "results.h5").read_bytes()).hexdigest(), data_hash)
        with h5py.File(new / "results.h5", "r") as result:
            np.testing.assert_allclose(result["Omega"][:], original, rtol=1e-12, atol=1e-12)

    def test_configuration_and_source_changes_do_not_reuse(self):
        original = self.run_case()
        self.assertNotEqual(self.run_case(self.config("--gamma", "500")), original)
        self.assertNotEqual(self.run_case(self.config("--N", "16")), original)
        self.assertNotEqual(self.run_case(self.config("--T", "0.006")), original)
        identity = bursting.identity_for(self.config()["parameters"])
        identity["source_sha256"]["solver/ns_periodic_mrSAV_solver.py"] = "changed-source"
        with patch.object(bursting, "identity_for", return_value=identity):
            self.assertNotEqual(self.run_case(), original)

    def test_adaptive_output_and_active_tolerances(self):
        config = self.config("--mode", "adaptive", "--rtol-q", "0.1")
        path = self.run_case(config)
        with h5py.File(path / "results.h5", "r") as result:
            n = len(result["tn"])
            self.assertEqual(len(result["tau"]), n-1)
            for key in ("rel_err", "controller_err", "ref_err", "q"):
                self.assertEqual(result[key].shape, (n,))
                self.assertTrue(np.isfinite(result[key][:]).all())
            self.assertEqual(result.attrs["snapshot_policy"], "linear_interpolation")
            self.assertFalse(result.attrs["compute_ref_err"])
            self.assertAlmostEqual(result["tn"][-1], .004)
        self.assertNotEqual(self.run_case(self.config("--mode", "adaptive", "--rtol-q", "0.2")), path)
        changed_execution = copy.deepcopy(config)
        changed_execution["execution"]["log_interval"] = 5
        self.assertEqual(self.run_case(changed_execution), path)

    def test_failure_and_interrupt_preserve_log_then_restart(self):
        for failure in (RuntimeError("injected solver failure"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(bursting, "integrate", side_effect=failure):
                    with self.assertRaises(type(failure)):
                        self.run_case(rerun=True)
        directories = list((self.root / "runs").iterdir())
        for directory in directories:
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("FAILED", (directory / "run.log").read_text())
            self.assertFalse((directory / "results.h5").exists())
        new = self.run_case()
        self.assertNotIn(new, directories)
        self.assertEqual(json.loads((new / "manifest.json").read_text())["status"], "completed")

    def test_unfinished_record_is_not_reused(self):
        old = self.run_case()
        path = old / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["status"] = "running"
        path.write_text(json.dumps(manifest))
        self.assertNotEqual(self.run_case(), old)
        self.assertEqual(json.loads(path.read_text())["status"], "running")

    def test_nonfinite_values_are_saved_and_failed_not_reused(self):
        original = bursting.integrate
        def nonfinite(*args):
            solver = original(*args)
            solver.Omega[-1, 0, 0] = np.nan
            warnings.warn("injected numerical warning", RuntimeWarning)
            return solver
        with patch.object(bursting, "integrate", side_effect=nonfinite):
            with self.assertRaises(FloatingPointError):
                self.run_case()
        directory = next((self.root / "runs").iterdir())
        self.assertEqual(json.loads((directory / "manifest.json").read_text())["status"], "failed")
        self.assertIn("injected numerical warning", (directory / "run.log").read_text())
        with h5py.File(directory / "results.h5", "r") as result:
            self.assertEqual(result.attrs["status"], "nonfinite")
            self.assertTrue(np.isnan(result["Omega"][-1, 0, 0]))
        self.assertNotEqual(self.run_case(), directory)

    def test_save_failure_cannot_mark_completion(self):
        with patch.object(bursting, "save_results", side_effect=OSError("injected disk error")):
            with self.assertRaises(OSError):
                self.run_case()
        directory = next((self.root / "runs").iterdir())
        self.assertEqual(json.loads((directory / "manifest.json").read_text())["status"], "failed")
        self.assertNotIn("COMPLETED", (directory / "run.log").read_text())

    def test_damaged_completed_record_requires_explicit_rerun(self):
        directory = self.run_case()
        with h5py.File(directory / "results.h5", "r+") as result:
            result.attrs["identity"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "Cannot reuse"):
            self.run_case()
        self.assertNotEqual(self.run_case(rerun=True), directory)

    def test_config_file_cli_priority_and_foreign_working_directory(self):
        source = self.root / "case.json"
        source.write_text(json.dumps({"Re": 50, "gamma": 500, "output_root": "runs/bursting"}))
        proc = subprocess.run([sys.executable, str(bursting.ROOT / "tests/legacy_bursting.py"),
                               "--config", str(source), "--gamma", "700", "--show-config"],
                              cwd=self.root, text=True, capture_output=True, check=True)
        resolved = json.loads(proc.stdout)
        self.assertEqual(resolved["parameters"]["Re"], 50)
        self.assertEqual(resolved["parameters"]["gamma"], 700)
        self.assertEqual(resolved["parameters"]["nu"], .02)
        self.assertEqual(Path(resolved["execution"]["output_root"]), bursting.ROOT / "runs/bursting")
        self.assertFalse((self.root / "runs").exists())

    def test_invalid_config_fails_before_running(self):
        for extra in (["--T", ".0041"], ["--N", "31"], ["--mode", "adaptive", "--tau-min", ".02"],
                      ["--mode", "adaptive", "--M", "IMEX"], ["--gamma", "nan"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.config(*extra)

    def test_progress_is_rate_limited_and_final_line_is_flushed(self):
        output = io.StringIO()
        logger = logging.Logger("progress-test")
        logger.addHandler(logging.StreamHandler(output))
        with patch.object(bursting, "monotonic", return_value=0):
            progress = bursting.ProgressLog(logger, 30, "main")
            for step in range(1000):
                progress.write(f"\r t={step}")
            self.assertEqual(len(output.getvalue().splitlines()), 1)
            progress.flush()
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertIn("t=999", output.getvalue())
        self.assertNotIn("\r", output.getvalue())
        progress.close()


if __name__ == "__main__":
    unittest.main()
