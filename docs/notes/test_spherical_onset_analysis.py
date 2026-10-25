"""Synthetic analysis validation: these fixtures are not simulation results."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

SCRIPT = Path(__file__).with_name("analyze_spherical_onset.py")
spec = importlib.util.spec_from_file_location("analysis", SCRIPT)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
REF = dict(cases=[dict(eta=0.5, ell=3, Ra=850.0)], planar=dict(Ra=696.0, k=2.25))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="snapy-synthetic-analysis-")
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def runfile(self, name, ra=850.0, nr=8, n=16, seed=1e-6, scale=8.0, end=0.2, rate=None):
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
        Path(str(p) + ".json").write_text(json.dumps(m))
        return p

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

    def test_coverage_requires_all_ratios_and_refinement(self):
        result = self.fixture()
        self.assertTrue(a.coverage(result, [0.5], 2, 1e-6, 0.8, "z", None)["complete"])
        self.assertFalse(a.coverage(result, [0.5, 0.6], 2, 1e-6, 0.8, "z", None)["complete"])
        self.assertFalse(a.coverage(result, [0.5], 2, 1e-6, 0.8, "z", 1)["complete"])

    def test_richardson_nonuniform_refinement(self):
        result = self.fixture()
        comparisons = a.refinement_comparisons(result["groups"])
        third = [c for c in comparisons if "observed_order" in c][0]
        self.assertAlmostEqual(third["observed_order"], 2, places=8)
        self.assertAlmostEqual(third["empirical_extrapolation"], 850, places=8)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
