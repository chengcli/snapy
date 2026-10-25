"""Synthetic analysis validation: these fixtures are not simulation results."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).with_name("analyze_spherical_onset.py")
spec = importlib.util.spec_from_file_location("analysis", SCRIPT)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
REF = dict(cases=[dict(eta=0.5, ell=3, Ra=850.0)], planar=dict(Ra=696.0, k=2.25))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="snapy-synthetic-analysis-", dir="/tmp")
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def runfile(
        self, name, ra=850.0, nr=8, n=16, seed=1e-6, scale=8.0, end=0.2, rate=None, dtype=None
    ):
        if rate is None:
            rate = (ra - (850 + 640 / nr**2)) * 0.01
        t = np.linspace(0, end, 201)
        amp = seed * np.exp(rate * t)
        data = np.column_stack((t / 0.02, t, amp, np.abs(amp), t * 1e-12, t * 1e-12))
        p = self.path / (name + ".csv")
        np.savetxt(
            p,
            data,
            delimiter=",",
            header="time,diffusive_time,amplitude,radial_rms,temperature_drift,density_drift",
            comments="",
        )
        m = dict(
            eta=0.5,
            ell=3,
            N=n,
            nr=nr,
            a=1,
            D=0.5,
            gamma=5 / 3,
            Pr=1,
            seed=seed,
            cfl=0.8,
            diffusion_step_scale=scale,
            orientation="z",
            eigenmode_seed=True,
            Ra=ra,
            device="cuda:0",
            end_diffusive_time=end,
            cycles=200,
            seconds=1.0,
            retries=0,
            chi_mid=0.02,
        )
        if dtype is not None:
            m["dtype"] = dtype
        Path(str(p) + ".json").write_text(json.dumps(m))
        return p

    def rewrite_csv(self, path, data):
        np.savetxt(
            path,
            data,
            delimiter=",",
            header="time,diffusive_time,amplitude,radial_rms,temperature_drift,density_drift",
            comments="",
        )

    def test_roundoff_final_duplicate_coalesced(self):
        for delta in (0.0, -np.finfo(float).eps):
            with self.subTest(delta=delta):
                p = self.runfile("duplicate", rate=-0.25)
                data = np.loadtxt(p, delimiter=",", skiprows=1)
                final = data[-1].copy()
                final[:2] += delta
                final[2] *= 1 + 1e-12
                self.rewrite_csv(p, np.vstack((data, final)))
                cleaned, meta = a.read_run(p)
                self.assertEqual(len(cleaned), len(data))
                self.assertEqual(meta["roundoff_endpoint_rows_removed"], 1)
                self.assertEqual(cleaned["amplitude"][-1], final[2])
                self.assertAlmostEqual(a.fit(cleaned, (0.15, 0.2))["growth"], -0.25, places=8)

    def test_final_physical_time_duplicate_coalesced(self):
        p = self.runfile("physical-duplicate")
        data = np.loadtxt(p, delimiter=",", skiprows=1)
        final = data[-1].copy()
        final[1] = np.nextafter(final[1], np.inf)
        self.rewrite_csv(p, np.vstack((data, final)))
        cleaned, meta = a.read_run(p)
        self.assertEqual(len(cleaned), len(data))
        self.assertEqual(meta["roundoff_endpoint_rows_removed"], 1)

    def test_non_roundoff_duplicates_rejected(self):
        for scenario in ("internal", "reversal", "different-state", "different-time"):
            with self.subTest(scenario=scenario):
                p = self.runfile(scenario)
                data = np.loadtxt(p, delimiter=",", skiprows=1)
                if scenario == "internal":
                    data = np.insert(data, 50, data[50], axis=0)
                else:
                    final = data[-1].copy()
                    if scenario == "reversal":
                        final[:2] -= 1e-5
                    elif scenario == "different-state":
                        final[2] *= 2
                    else:
                        final[0] += 1e-5
                    data = np.vstack((data, final))
                self.rewrite_csv(p, data)
                with self.assertRaisesRegex(ValueError, "Nonmonotone"):
                    a.read_run(p)

    def test_exact_growth_and_windows(self):
        p = self.runfile("growth", rate=-0.25)
        d, m = a.read_run(p)
        fits = [a.fit(d, w) for w in a.windows_for(0.2)]
        self.assertTrue(np.allclose([f["growth"] for f in fits], -0.25, atol=1e-12))
        self.assertTrue(
            np.allclose(a.windows_for(0.1), [(0.025, 0.05), (0.05, 0.075), (0.075, 0.1)])
        )

    def test_linear_quadratic_brackets(self):
        r = a.neutral_root([800, 900], [-0.5, 0.5])
        self.assertAlmostEqual(r["Ra"], 850)
        xs = np.array([800, 850, 900])
        ys = (xs - 860) * 0.01 + 0.00001 * (xs - 860) ** 2
        r = a.neutral_root(xs, ys)
        self.assertAlmostEqual(r["Ra"], 860)
        self.assertIn("quadratic", r["method"])

    def test_no_bracket_and_duplicates_rejected(self):
        for x, y in [
            ([800, 900], [0.1, 0.2]),
            ([800, 800, 900], [-0.1, -0.1, 0.1]),
            ([800, 850, 900], [-0.1, 0.2, 0.1]),
        ]:
            with self.assertRaises(ValueError):
                a.neutral_root(x, y)

    def test_invalid_gpu_retry_and_time(self):
        for field, value in [
            ("device", "cpu"),
            ("retries", 1),
            ("a", 2),
            ("end_diffusive_time", 0.3),
        ]:
            p = self.runfile(field)
            mp = Path(str(p) + ".json")
            m = json.loads(mp.read_text())
            m[field] = value
            mp.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                a.read_run(p)

    def test_nonfinite_and_insufficient_fits(self):
        p = self.runfile("finite")
        d, m = a.read_run(p)
        with self.assertRaises(ValueError):
            a.fit(d, (0.01, 0.011), 10)
        d["amplitude"][90] = 0
        with self.assertRaises(ValueError):
            a.fit(d, (0.05, 0.1))
        d["time"][5] = np.nan
        np.savetxt(
            p,
            np.column_stack([d[k] for k in d.dtype.names]),
            delimiter=",",
            header=",".join(d.dtype.names),
            comments="",
        )
        with self.assertRaises(ValueError):
            a.read_run(p)

    def test_incomplete_skipped(self):
        p = self.runfile("pending")
        Path(str(p) + ".json").unlink()
        result = a.analyze(self.path, REF)
        self.assertEqual(result["incomplete_runs"], ["pending.csv"])
        self.assertFalse(result["runs"])

    def fixture(self):
        for nr in [8, 12, 16]:
            for ra in [800, 850, 900]:
                self.runfile(f"n{nr}-ra{ra}", ra, nr, 2 * nr)
        self.runfile("dtcontrol", 900, 12, 24, scale=1)
        self.runfile("seedcontrol", 900, 12, 24, seed=1e-7)
        self.runfile("rest", 900, 12, 24, seed=0)
        return a.analyze(self.path, REF)

    def test_grouping_and_controls(self):
        result = self.fixture()
        self.assertEqual(sum("neutral" in g for g in result["groups"]), 3)
        self.assertEqual(len(result["rest_controls"]), 1)
        self.assertEqual({c["kind"] for c in result["controls"]}, {"seed", "timestep"})
        self.assertLess(
            max(c["max_seed_normalized_amplitude_difference"] for c in result["controls"]), 1e-12
        )

    def test_precision_groups_and_controls_are_separate(self):
        for dtype, root in (("float32", 875), ("float64", 850)):
            for ra in (800, 850, 900):
                self.runfile(f"{dtype}-{ra}", ra=ra, rate=(ra - root) * 0.01, dtype=dtype)
        result = a.analyze(self.path, REF)
        self.assertFalse(result["invalid_runs"])
        self.assertEqual(len(result["groups"]), 2)
        roots = {g["parameters"]["dtype"]: g["neutral"]["Ra"] for g in result["groups"]}
        self.assertAlmostEqual(roots["float32"], 875)
        self.assertAlmostEqual(roots["float64"], 850)
        self.assertEqual(len(result["controls"]), 3)
        for control in result["controls"]:
            self.assertEqual(control["kind"], "precision")
            self.assertEqual(set(control["growth_by_dtype"]), {"float32", "float64"})
        self.assertFalse(a.refinement_comparisons(result["groups"]))

    def test_float32_cannot_supply_baseline_refinement_or_step_scale(self):
        for ra in (800, 900):
            self.runfile(f"legacy-{ra}", ra=ra)
            self.runfile(f"single-{ra}", ra=ra, nr=12, n=24, dtype="float32")
            for nr in (16, 24):
                self.runfile(
                    f"single-scale-{nr}-{ra}", ra=ra, nr=nr, n=nr * 2, scale=1, dtype="float32"
                )
        result = a.analyze(self.path, REF)
        self.assertTrue(
            all(r["dtype"] == "float64" for r in result["runs"] if r["name"].startswith("legacy"))
        )
        cov = a.coverage(result, [0.5], 2, 1e-6, 0.8, "z", None)
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["baseline"]["dtype"], "float64")
        self.assertEqual(cov["baseline"]["diffusion_step_scale"], 8)
        self.assertEqual(cov["ratios"][0]["grids"], [dict(nr=8, N=16)])
        for ra in (800, 900):
            self.runfile(f"double-fine-{ra}", ra=ra, nr=12, n=24, dtype="float64")
        result = a.analyze(self.path, REF)
        self.assertTrue(a.coverage(result, [0.5], 2, 1e-6, 0.8, "z", None)["complete"])

    def test_coverage_requires_all_ratios_and_refinement(self):
        result = self.fixture()
        self.assertTrue(a.coverage(result, [0.5], 2, 1e-6, 0.8, "z", None)["complete"])
        self.assertFalse(a.coverage(result, [0.5, 0.6], 2, 1e-6, 0.8, "z", None)["complete"])
        self.assertFalse(a.coverage(result, [0.5], 2, 1e-6, 0.8, "z", 1)["complete"])

    def test_strict_cli_defaults_reject_partial_without_writing_artifacts(self):
        self.fixture()
        reference = self.path / "reference.json"
        reference.write_text(json.dumps(REF))
        output = self.path / "strict-output"
        run = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPT),
                "--input-dir",
                str(self.path),
                "--reference",
                str(reference),
                "--output-dir",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(run.returncode, 1)
        diagnostic = json.loads(run.stdout)
        self.assertEqual(diagnostic["status"], "partial_diagnostic")
        self.assertEqual(diagnostic["coverage"]["required_ratios"], list(a.ETAS))
        self.assertEqual(diagnostic["coverage"]["minimum_refined_grids"], 2)
        self.assertIn("no artifacts written", run.stderr)
        self.assertFalse(output.exists())

    def test_all_eight_ratios_need_two_float64_grids_and_completed_inputs(self):
        result = self.fixture()
        templates = [g for g in result["groups"] if "neutral" in g][:2]
        result["groups"] = []
        self.assertEqual(len(a.ETAS), 8)
        for eta in a.ETAS:
            for template in templates:
                group = copy.deepcopy(template)
                group["parameters"]["eta"] = eta
                result["groups"].append(group)

        def check():
            return a.coverage(result, a.ETAS, 2, 1e-6, 0.8, "z", 8)

        self.assertTrue(check()["complete"])
        result["groups"][-1]["parameters"]["dtype"] = "float32"
        self.assertFalse(check()["complete"])
        self.assertEqual(check()["missing_or_unrefined_ratios"], [0.99])
        result["groups"][-1]["parameters"]["dtype"] = "float64"
        result["incomplete_runs"].append("running.csv")
        self.assertFalse(check()["complete"])

    def test_richardson_nonuniform_refinement(self):
        result = self.fixture()
        comparisons = a.refinement_comparisons(result["groups"])
        third = [c for c in comparisons if "observed_order" in c][0]
        self.assertEqual(third["order_fit_status"], "fitted")
        self.assertAlmostEqual(third["observed_order"], 2, places=8)
        self.assertAlmostEqual(third["empirical_extrapolation"], 850, places=8)

    def test_failed_three_grid_order_is_recorded_and_reported(self):
        from docutils.core import publish_doctree

        result = self.fixture()
        groups = sorted(
            [g for g in result["groups"] if "neutral" in g],
            key=lambda g: g["parameters"]["nr"],
        )
        cases = [
            ([700.974, 704.093, 708.710], "No positive-order fit"),
            ([700.974, 704.093, 702.000], "Sign-changing neutral-Ra increments"),
            ([700.974, 704.093, 704.093], "Zero neutral-Ra increment"),
        ]
        for values, reason in cases:
            with self.subTest(values=values):
                for group, ra in zip(groups, values):
                    group["parameters"].update(eta=0.9, N=6 * group["parameters"]["nr"])
                    group["neutral"]["Ra"] = ra
                    group["neutral"]["relative_error"] = ra / group["reference"]["modal_Ra"] - 1
                comparisons = a.refinement_comparisons(groups)
                third = next(c for c in comparisons if "order_fit_status" in c)
                self.assertEqual(third["order_fit_status"], "failed")
                self.assertIn(reason, third["reason"])
                self.assertEqual(third["Ra_values"], values)
                self.assertEqual(third["grids"], [dict(nr=n, N=6 * n) for n in (8, 12, 16)])
                np.testing.assert_allclose(third["neutral_increments"], np.diff(values))
                self.assertNotIn("observed_order", third)
                self.assertNotIn("empirical_extrapolation", third)
                result["refinement_comparisons"] = comparisons
                result["coverage"] = a.coverage(result, a.ETAS, 2, 1e-6, 0.8, "z", 8)
                result.update(total_cycles=0, total_gpu_run_seconds=0.0)
                output = self.path / "failed-order-output"
                a.write_artifacts(result, REF, output)
                summary = (output / "spherical_onset_summary.inc").read_text()
                self.assertIn("Three-grid order fit failed at eta=0.9", summary)
                self.assertIn(reason, summary)
                for n, ra in zip((8, 12, 16), values):
                    self.assertIn(f"nr/N={n}/{6 * n}: Ra={ra:.6f}", summary)
                self.assertIn("not an observed convergence result", summary)
                saved = json.loads((output / "spherical_onset_results.json").read_text())
                self.assertIn(third, saved["refinement_comparisons"])
                publish_doctree(summary, settings_overrides={"halt_level": 2, "report_level": 2})

    def test_three_grid_order_reports_incompatible_grids_and_windows(self):
        result = self.fixture()
        groups = sorted(
            [g for g in result["groups"] if "neutral" in g],
            key=lambda g: g["parameters"]["nr"],
        )
        for scenario, reason in (("grid", "in proportion"), ("window", "different fit windows")):
            with self.subTest(scenario=scenario):
                modified = copy.deepcopy(groups)
                if scenario == "grid":
                    modified[-1]["parameters"]["N"] += 8
                else:
                    modified[-1]["neutral"]["window"] = [0.1, 0.15]
                third = next(
                    c for c in a.refinement_comparisons(modified) if "order_fit_status" in c
                )
                self.assertEqual(third["order_fit_status"], "not_applicable")
                self.assertIn(reason, third["reason"])
                self.assertNotIn("observed_order", third)

    def test_window_sensitivity_recorded(self):
        result = self.fixture()
        self.assertTrue(
            all(abs(r.get("last_window_growth_shift", 0)) < 1e-10 for r in result["runs"])
        )
        self.assertTrue(all(abs(g.get("last_window_Ra_shift", 0)) < 1e-7 for g in result["groups"]))

    def test_artifacts_and_rst(self):
        from docutils.core import publish_doctree

        result = self.fixture()
        result["coverage"] = a.coverage(result, [0.5, 0.6], 2, 1e-6, 0.8, "z", 8)
        result.update(total_cycles=2400, total_gpu_run_seconds=12.0)
        out = self.path / "synthetic-output"
        a.write_artifacts(result, REF, out)
        self.assertIn("PARTIAL DIAGNOSTIC", (out / "spherical_onset_summary.inc").read_text())
        self.assertTrue((out / "figures/spherical_compressible_onset.png").stat().st_size > 1000)
        for name in ["spherical_onset_tables.inc", "spherical_onset_summary.inc"]:
            publish_doctree(
                (out / name).read_text(), settings_overrides={"halt_level": 2, "report_level": 2}
            )

    def test_plot_and_summary_keep_precision_controls_out_of_baseline(self):
        result = self.fixture()
        for ra in (800, 850, 900):
            self.runfile(f"float32-{ra}", ra=ra, dtype="float32", rate=(ra - 890) * 0.01)
        self.runfile("precision32", seed=0.01, dtype="float32", rate=-0.02)
        self.runfile("precision64", seed=0.01, dtype="float64", rate=0.06)
        result = a.analyze(self.path, REF)
        result["coverage"] = a.coverage(result, a.ETAS, 2, 1e-6, 0.8, "z", 8)
        result.update(total_cycles=0, total_gpu_run_seconds=0.0)
        out = self.path / "precision-output"
        with patch.object(a.plt, "close"):
            a.write_artifacts(result, REF, out)
            fig = a.plt.gcf()
        try:
            axes = fig.axes
            self.assertEqual(len(axes[2].collections), 3)
            self.assertIn("Numerical error", axes[2].get_ylabel())
            expected = [
                g
                for g in result["groups"]
                if "neutral" in g and a.matches_baseline(g, result["coverage"]["baseline"])
            ]
            expected.sort(key=lambda g: g["parameters"]["nr"])
            for points, group in zip(axes[2].collections, expected):
                np.testing.assert_allclose(
                    points.get_offsets(), [[0.5, 100 * group["neutral"]["relative_error"]]]
                )
            self.assertTrue(
                all("seed=" not in text.get_text() for text in axes[0].get_legend().get_texts())
            )
        finally:
            a.plt.close(fig)
        summary = (out / "spherical_onset_summary.inc").read_text()
        self.assertIn("PARTIAL DIAGNOSTIC", summary)
        self.assertIn("float32 growth -0.02; float64 growth +0.06", summary)
        self.assertIn("absolute growth-rate difference 0.08", summary)
        self.assertIn("disagree on growth versus decay", summary)
        self.assertIn("-133.333%", summary)
        self.assertIn("float32", (out / "spherical_onset_tables.inc").read_text())
        from docutils.core import publish_doctree

        publish_doctree(summary, settings_overrides={"halt_level": 2, "report_level": 2})

    def test_near_zero_precision_growth_keeps_absolute_difference_primary(self):
        self.runfile("nearzero32", dtype="float32", rate=1e-8)
        self.runfile("nearzero64", dtype="float64", rate=1e-12)
        result = a.analyze(self.path, REF)
        control = result["controls"][0]
        self.assertAlmostEqual(control["absolute_growth_difference"], 1e-8 - 1e-12, delta=1e-11)
        result["coverage"] = a.coverage(result, a.ETAS, 2, 1e-6, 0.8, "z", 8)
        result.update(total_cycles=0, total_gpu_run_seconds=0.0)
        output = self.path / "nearzero-output"
        a.write_artifacts(result, REF, output)
        summary = (output / "spherical_onset_summary.inc").read_text()
        self.assertIn("amplified by a near-zero float64 growth rate", summary)
        self.assertLess(
            summary.index("absolute growth-rate difference"),
            summary.index("Supplementary relative"),
        )

    def test_same_eta_angular_controls_remain_unconnected_grid_points(self):
        reference = dict(cases=[dict(eta=0.99, ell=224, Ra=697.0)], planar=REF["planar"])
        roots = [706.868, 702.0]
        for n, root in zip((336, 456), roots):
            for ra in (680, 700, 740):
                path = self.runfile(f"n{n}-{ra}", ra=ra, nr=8, n=n, rate=0.01 * (ra - root))
                sidecar = Path(str(path) + ".json")
                meta = json.loads(sidecar.read_text())
                meta.update(eta=0.99, ell=224)
                sidecar.write_text(json.dumps(meta))
        result = a.analyze(self.path, reference)
        result["coverage"] = a.coverage(result, a.ETAS, 2, 1e-6, 0.8, "z", 8)
        result.update(total_cycles=0, total_gpu_run_seconds=0.0)
        out = self.path / "angular-controls-output"
        with patch.object(a.plt, "close"):
            a.write_artifacts(result, reference, out)
            fig = a.plt.gcf()
        try:
            np.testing.assert_allclose(
                fig.axes[0].collections[0].get_offsets(), [[0.99, ra] for ra in roots]
            )
            np.testing.assert_allclose(
                fig.axes[2].collections[0].get_offsets(),
                [[0.99, 100 * (ra / 697.0 - 1)] for ra in roots],
            )
            self.assertEqual(len(fig.axes[0].lines), 2)  # Spectral curve and planar limit only.
            self.assertEqual(len(fig.axes[2].lines), 1)  # Zero-error reference only.
        finally:
            a.plt.close(fig)
        self.assertIn("unconnected markers", (out / "spherical_onset_summary.inc").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
