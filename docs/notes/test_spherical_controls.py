"""Synthetic sidecar-report tests; fixtures are not simulation results."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_spherical_controls as controls
import numpy as np


class ControlStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="snapy-control-tests-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.samples = self.path / "samples"
        self.samples.mkdir()
        self.reference = dict(cases=[], planar=dict(Ra=696.0))
        self.reference_path = self.path / "reference.json"
        self.reference_path.write_text(json.dumps(self.reference))
        self.counter = 0
        self.modal_reference = patch.object(
            controls.onset,
            "reference_for",
            side_effect=lambda reference, meta: dict(
                modal_Ra=self.spectral(meta["eta"], meta["ell"])
            ),
        )
        self.modal_reference.start()
        self.addCleanup(self.modal_reference.stop)

    @staticmethod
    def spectral(eta, ell):
        return 701.609 if eta == 0.95 else {20: 710.0, 21: 708.0, 22: 712.0}[ell]

    def runfile(self, directory, eta, ell, orientation, end, factor, shift=0, late_drift=0):
        self.counter += 1
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"unrelated-name-{self.counter}.csv"
        reference = self.spectral(eta, ell)
        ra = factor * reference
        t = np.linspace(0, end, round(end * 1000) + 1)
        log_amp = 0.01 * (
            (ra - reference - shift) * t - 0.5 * late_drift * np.maximum(t - 0.2, 0) ** 2
        )
        amp = 1e-6 * np.exp(log_amp)
        np.savetxt(
            path,
            np.column_stack((t / 0.02, t, amp, np.abs(amp), t * 1e-12, t * 1e-12)),
            delimiter=",",
            comments="",
            header="time,diffusive_time,amplitude,radial_rms,temperature_drift,density_drift",
        )
        meta = dict(
            controls.COMMON,
            eta=eta,
            ell=ell,
            N=132 if eta == 0.95 else 72,
            orientation=orientation,
            end_diffusive_time=end,
            Ra=ra,
            device="cuda:0",
            chi_mid=0.02,
            cycles=len(t) - 1,
            seconds=1.0,
            retries=0,
        )
        Path(str(path) + ".json").write_text(json.dumps(meta))
        return path

    def fixture(self, with_long=True):
        paths = dict(short=[], long=[], modes=[])
        for factor in controls.FACTORS:
            paths["short"].append(self.runfile(self.samples, 0.95, 43, "z", 0.2, factor))
            if with_long:
                paths["long"].append(
                    self.runfile(
                        self.samples / "long_window",
                        0.95,
                        43,
                        "z",
                        0.6,
                        factor,
                        late_drift=10,
                    )
                )
        for ell, axis, shift in (
            (20, "z", -1),
            (21, "z", -2),
            (22, "z", -3),
            (21, "diagonal", 0.5),
        ):
            paths["modes"].append(
                [
                    self.runfile(self.samples, 0.9, ell, axis, 0.2, factor, shift=shift)
                    for factor in controls.FACTORS
                ]
            )
        return paths

    def analyze(self):
        return controls.analyze(self.samples, self.reference)

    def change_meta(self, path, **changes):
        sidecar = Path(str(path) + ".json")
        meta = json.loads(sidecar.read_text())
        meta.update(changes)
        sidecar.write_text(json.dumps(meta))

    def test_complete_windows_repeat_and_modes(self):
        self.fixture()
        result = self.analyze()
        self.assertEqual(result["status"], "control_coverage_complete")
        self.assertEqual(
            [w["window"] for w in result["long"]["windows"]],
            [list(w) for w in controls.LONG_WINDOWS],
        )
        for row in result["long"]["windows"]:
            expected = 701.609 + 10 * max(np.mean(row["window"]) - 0.2, 0)
            self.assertAlmostEqual(row["neutral"]["Ra"], expected, places=7)
            self.assertEqual(len(row["fits"]), 3)
        for comparison in result["short_window_repeats"]:
            self.assertAlmostEqual(comparison["neutral_difference"], 0, places=8)
            self.assertTrue(
                all(abs(f["difference"]) < 1e-10 for f in comparison["growth_differences"])
            )
        modes = result["mode_controls"]
        np.testing.assert_allclose(
            [m["windows"][0]["neutral"]["Ra"] for m in modes], [709, 706, 709, 708.5]
        )
        self.assertAlmostEqual(modes[-1]["neutral_difference_from_ell21_z"], 2.5)

    def test_missing_long_directory_is_partial_with_original_roots(self):
        self.fixture(with_long=False)
        result = self.analyze()
        self.assertEqual(result["status"], "partial_diagnostic")
        self.assertEqual(result["long"]["matched_count"], 0)
        self.assertTrue(all(w["neutral"] is None for w in result["long"]["windows"]))
        self.assertTrue(all(w["neutral"] is not None for w in result["short"]["windows"]))
        self.assertFalse((self.samples / "long_window").exists())
        out = self.path / "preview"
        controls.write_artifacts(result, out)
        self.assertIn("PARTIAL DIAGNOSTIC", (out / "spherical_control_study.inc").read_text())
        self.assertGreater((out / "figures/spherical_control_study.png").stat().st_size, 1000)

    def test_two_completed_long_runs_never_form_a_triplet(self):
        paths = self.fixture()
        Path(str(paths["long"][1]) + ".json").unlink()
        result = self.analyze()
        self.assertEqual(result["long"]["matched_count"], 2)
        self.assertEqual(len(result["input_audit"]["long"]["incomplete"]), 1)
        self.assertTrue(
            all(not w["fits"] and w["neutral"] is None for w in result["long"]["windows"])
        )
        self.assertEqual(result["status"], "partial_diagnostic")

    def test_duplicate_completed_repeat_is_ambiguous(self):
        self.fixture()
        self.runfile(self.samples / "long_window", 0.95, 43, "z", 0.6, 1.0)
        result = self.analyze()
        self.assertEqual(result["status"], "partial_diagnostic")
        self.assertIn("found 2", result["long"]["issues"][0])
        self.assertTrue(all(w["neutral"] is None for w in result["long"]["windows"]))

    def test_precision_parameters_and_duration_cannot_be_mixed(self):
        paths = self.fixture()
        path = paths["long"][1]
        sidecar = Path(str(path) + ".json")
        original = sidecar.read_text()
        for changes in (
            dict(dtype="float32"),
            dict(seed=0.01),
            dict(diffusion_step_scale=4),
            dict(N=88),
        ):
            with self.subTest(changes=changes):
                self.change_meta(path, **changes)
                result = self.analyze()
                self.assertEqual(result["status"], "partial_diagnostic")
                self.assertEqual(result["long"]["matched_count"], 2)
                sidecar.write_text(original)
        # A valid 0.2 repeat placed in long_window cannot substitute for a 0.6 run.
        path.unlink()
        sidecar.unlink()
        self.runfile(self.samples / "long_window", 0.95, 43, "z", 0.2, 1.0)
        self.assertEqual(self.analyze()["long"]["matched_count"], 2)

    def test_invalid_required_sample_and_missing_mode_keep_partial_status(self):
        paths = self.fixture()
        self.change_meta(paths["long"][0], retries=1)
        Path(str(paths["modes"][2][0]) + ".json").unlink()
        result = self.analyze()
        self.assertEqual(result["status"], "partial_diagnostic")
        self.assertEqual(len(result["input_audit"]["long"]["invalid"]), 1)
        self.assertIn("retries", result["input_audit"]["long"]["invalid"][0]["reason"])
        self.assertEqual(result["mode_controls"][2]["status"], "partial_diagnostic")

    def test_late_windows_without_sign_bracket_are_not_extrapolated(self):
        self.fixture(with_long=False)
        for factor in controls.FACTORS:
            self.runfile(self.samples / "long_window", 0.95, 43, "z", 0.6, factor, late_drift=1000)
        result = self.analyze()
        self.assertEqual(result["status"], "partial_diagnostic")
        self.assertEqual(result["long"]["matched_count"], 3)
        self.assertEqual(result["long"]["windows"][0]["status"], "complete")
        last = result["long"]["windows"][-1]
        self.assertIsNone(last["neutral"])
        self.assertEqual(len(last["fits"]), 3)
        self.assertIn("No measured decay/growth bracket", last["issues"][0])

    def test_artifacts_and_rst(self):
        from docutils.core import publish_doctree

        self.fixture()
        result = self.analyze()
        out = self.path / "complete-preview"
        with patch.object(controls.plt, "close"):
            controls.write_artifacts(result, out)
            figure = controls.plt.gcf()
        try:
            times = [np.mean(window) for window in controls.LONG_WINDOWS]
            for index, line in enumerate(figure.axes[0].lines[:3]):
                np.testing.assert_allclose(line.get_xdata(), times)
                np.testing.assert_allclose(
                    line.get_ydata(),
                    [window["fits"][index]["growth"] for window in result["long"]["windows"]],
                )
            np.testing.assert_allclose(
                np.asarray(figure.axes[1].lines[0].get_ydata(), dtype=float),
                [window["neutral"]["Ra"] for window in result["long"]["windows"]],
            )
        finally:
            controls.plt.close(figure)
        include = out / "spherical_control_study.inc"
        summary = include.read_text()
        self.assertIn("Supplemental control coverage complete", summary)
        self.assertIn("0.45--0.6", summary)
        self.assertIn("21 / diagonal", summary)
        self.assertIn("figures/spherical_control_study.png", summary)
        saved = json.loads((out / "spherical_control_study.json").read_text())
        self.assertEqual(saved["status"], "control_coverage_complete")
        self.assertEqual(len(saved["long"]["windows"]), 5)
        self.assertIn(saved["repeat_sampling_note"], summary)
        publish_doctree(
            summary,
            source_path=str(include),
            settings_overrides={"halt_level": 2, "report_level": 2},
        )

    def test_strict_cli_refuses_partial_output(self):
        self.fixture(with_long=False)
        output = self.path / "strict-output"
        argv = [
            str(Path(controls.__file__)),
            "--input-dir",
            str(self.samples),
            "--reference",
            str(self.reference_path),
            "--output-dir",
            str(output),
        ]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SystemExit, "no artifacts written"):
                controls.main()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
