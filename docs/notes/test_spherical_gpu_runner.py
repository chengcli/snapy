"""CPU-only runner checks; subprocess fixtures own all processes they signal."""

import contextlib
import ctypes
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import call, patch

import run_spherical_gpu as runner
from study_archive import read_data

REFERENCE = {"cases": [{"eta": 0.5, "ell": 3, "Ra": 850.0}]}


class PlanningTests(unittest.TestCase):
    def test_cli_degree(self):
        self.assertIsNone(runner.parse_args([]).ell)
        self.assertEqual(runner.parse_args(["--ell", "21"]).ell, 21)
        for value in ("0", "-1", "2.5", "nan", "abc"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.parse_args(["--ell", value])
                self.assertEqual(error.exception.code, 2)

    def test_default_plan_preserved(self):
        args = runner.parse_args(["--etas", ".5"])
        with patch.object(runner, "neutral") as neutral:
            runs = runner.plan_runs(args, REFERENCE)
        neutral.assert_not_called()
        self.assertEqual(
            [r["name"] for r in runs],
            [f"eta050-n16-r8-scan-f{f}" for f in ("0940", "1000", "1060")],
        )
        self.assertEqual([r["Ra"] for r in runs], [f * 850 for f in (.94, 1, 1.06)])
        self.assertEqual(
            runs[0],
            dict(
                name="eta050-n16-r8-scan-f0940",
                eta=.5, ell=3, N=16, nr=8, Ra=.94 * 850,
                end=.2, seed=1e-6, cfl=.8, diffusion_step_scale=1.,
                dtype="float64", orientation="z",
            ),
        )

    def test_override_threshold_once_per_eta_and_degree_resolution(self):
        args = runner.parse_args(["--etas", ".5,.6", "--ell", "21"])
        with patch.object(runner, "neutral", side_effect=[1000., 2000.]) as neutral:
            runs = runner.plan_runs(args, REFERENCE)
        self.assertEqual(neutral.call_args_list, [call(48, .5, 21), call(48, .6, 21)])
        self.assertEqual([r["Ra"] for r in runs], [940., 1000., 1060., 1880., 2000., 2120.])
        self.assertTrue(all(r["ell"] == 21 and r["N"] == 48 for r in runs))
        self.assertEqual(runs[0]["name"], "eta050-ell21-n48-r8-scan-f0940")

    def test_explicit_critical_degree_still_has_distinct_filename(self):
        default = runner.plan_runs(runner.parse_args(["--etas", ".5"]), REFERENCE)
        with patch.object(runner, "neutral", return_value=850.):
            explicit = runner.plan_runs(
                runner.parse_args(["--etas", ".5", "--ell", "3"]), REFERENCE,
            )
        self.assertTrue({r["name"] for r in default}.isdisjoint(r["name"] for r in explicit))

    def test_real_same_degree_neutral_reference(self):
        reference = json.loads(read_data(runner.HERE / "spherical_reference_results.json"))
        args = runner.parse_args(["--etas", ".5", "--ell", "4", "--factors", ".94,1,1.06"])
        expected = runner.neutral(48, .5, 4)
        runs = runner.plan_runs(args, reference)
        self.assertGreater(expected, next(c["Ra"] for c in reference["cases"] if c["eta"] == .5))
        self.assertEqual([r["Ra"] for r in runs], [f * expected for f in (.94, 1, 1.06)])

    def test_resolution_scale_validation_preserved(self):
        args = runner.parse_args(["--etas", ".5", "--angular-scale", "1.1"])
        with self.assertRaisesRegex(ValueError, "even integral"):
            runner.plan_runs(args, REFERENCE)


class EnvironmentTests(unittest.TestCase):
    def test_distributed_preserves_caller_transport(self):
        caller = dict(
            UCX_TLS="self,tcp,cuda_copy", COMMUX_GROUP="0", COMMUX_COALESCE="0",
        )
        with patch.dict(os.environ, caller, clear=True):
            environment = runner.run_environment(2, "2,3")
            self.assertEqual(dict(os.environ), caller)
        self.assertEqual(environment["UCX_TLS"], caller["UCX_TLS"])
        self.assertEqual(environment["BACKEND"], "ucx")
        self.assertEqual(environment["DEVICE"], "cuda")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2,3")
        self.assertEqual(runner.transport_settings(environment, 2), caller)

    def test_distributed_default_transport(self):
        environment = runner.run_environment(2, "0,1", {})
        self.assertEqual(environment["UCX_TLS"], "self,tcp,cuda_copy,cuda_ipc")
        self.assertEqual(runner.transport_settings(environment, 2), dict(
            UCX_TLS="self,tcp,cuda_copy,cuda_ipc", COMMUX_GROUP="1", COMMUX_COALESCE="1",
        ))

    def test_single_process_environment_unchanged(self):
        caller = {"UCX_TLS": "self,tcp,cuda_copy", "COMMUX_GROUP": "0"}
        environment = runner.run_environment(1, "0", caller)
        self.assertEqual(environment, caller)
        self.assertIsNot(environment, caller)
        self.assertEqual(runner.run_environment(1, "0", {}), {})
        self.assertEqual(runner.transport_settings({}, 1), dict(
            UCX_TLS=None, COMMUX_GROUP=None, COMMUX_COALESCE=None,
        ))


# Each generation starts a new session, like torchrun's detached workers.
# Files are written only after descendants are ready, avoiding timing guesses.
TREE_SCRIPT = """
import os, pathlib, subprocess, sys, time
directory, depth = pathlib.Path(sys.argv[1]), int(sys.argv[2])
child = None
if depth:
    child = subprocess.Popen([sys.executable, __file__, str(directory), str(depth - 1)],
                             start_new_session=True)
    deadline = time.monotonic() + 5
    while not (directory / str(depth - 1)).exists():
        if time.monotonic() > deadline:
            raise RuntimeError('fixture descendant did not start')
        time.sleep(.01)
(directory / str(depth)).write_text(str(os.getpid()))
try:
    time.sleep(30)
finally:
    if child is not None:
        child.kill()
        child.wait()
"""


@unittest.skipUnless(sys.platform == "linux", "owned-tree cleanup uses Linux /proc")
class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="snapy-runner-test-")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        # Adopt and reap only our orphaned fixtures rather than leave zombies
        # to the host's PID 1. Restore the previous setting after each test.
        libc = ctypes.CDLL(None, use_errno=True)
        old = ctypes.c_int()
        if libc.prctl(37, ctypes.byref(old), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
            self.fail("Could not enable fixture child subreaper")
        self.addCleanup(libc.prctl, 36, old.value, 0, 0, 0)

    def reap(self, pid):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    return
            except ChildProcessError:
                if not Path(f"/proc/{pid}").exists():
                    return
            time.sleep(.01)
        self.fail(f"Owned fixture process {pid} survived cleanup")

    def exercise_tree_cleanup(self, failure=None):
        script = self.directory / "tree.py"
        script.write_text(TREE_SCRIPT)
        command = [sys.executable, str(script), str(self.directory), "2"]
        processes = []
        pids = []
        reaped = set()
        real_wait = subprocess.Popen.wait

        def wait(process, timeout=None):
            if not processes:
                processes.append(process)
                deadline = time.monotonic() + 5
                while not (self.directory / "2").exists():
                    if time.monotonic() > deadline:
                        raise AssertionError("Owned fixture tree did not become ready")
                    time.sleep(.01)
                pids.extend(int((self.directory / str(i)).read_text()) for i in range(3))
                self.assertEqual(pids[-1], process.pid)
                self.assertEqual(len({os.getsid(pid) for pid in pids}), 3)
                if failure is not None:
                    raise failure
            return real_wait(process, timeout=.02 if timeout is not None else 3)

        # A sibling in another session must remain untouched by tree cleanup.
        with subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
        ) as unrelated:
            try:
                expected_error = type(failure) if failure is not None else RuntimeError
                with (self.directory / "run.log").open("w") as log:
                    with patch.object(subprocess.Popen, "wait", wait):
                        with self.assertRaises(expected_error) as error:
                            runner.run_with_progress(
                                command, os.environ.copy(), log,
                                self.directory / "absent.csv", .04,
                            )
                if failure is not None:
                    self.assertIs(error.exception, failure)
                else:
                    self.assertIn("No CSV progress", str(error.exception))
                self.assertIsNotNone(processes[0].returncode)
                self.assertIsNone(unrelated.poll())
                for pid in pids:
                    self.reap(pid)
                    reaped.add(pid)
            finally:
                unrelated.kill()
                # Failures in the assertions must not leak our fixtures either.
                for depth in range(3):
                    path = self.directory / str(depth)
                    if path.exists():
                        pid = int(path.read_text())
                        if pid in reaped:
                            continue
                        if (
                            processes and pid == processes[0].pid
                            and processes[0].returncode is not None
                        ):
                            continue
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        self.reap(pid)
                for process in processes:
                    real_wait(process, timeout=3)

    def test_stall_kills_separate_session_descendants(self):
        self.exercise_tree_cleanup()

    def test_keyboard_interrupt_kills_descendants(self):
        self.exercise_tree_cleanup(KeyboardInterrupt("test interrupt"))

    def test_system_exit_kills_descendants(self):
        self.exercise_tree_cleanup(SystemExit(17))

    def test_base_exception_kills_descendants(self):
        self.exercise_tree_cleanup(BaseException("test cancellation"))

    def test_ordinary_error_kills_descendants(self):
        self.exercise_tree_cleanup(OSError("test progress failure"))

    def test_success_and_nonzero_exit(self):
        with (self.directory / "run.log").open("w") as log:
            for code in (0, 7):
                command = [sys.executable, "-c", f"raise SystemExit({code})"]
                if code:
                    with self.assertRaises(subprocess.CalledProcessError) as error:
                        runner.run_with_progress(
                            command, os.environ.copy(), log, self.directory / "none", 1,
                        )
                    self.assertEqual(error.exception.returncode, code)
                    self.assertEqual(error.exception.cmd, command)
                else:
                    runner.run_with_progress(
                        command, os.environ.copy(), log, self.directory / "none", 1,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
