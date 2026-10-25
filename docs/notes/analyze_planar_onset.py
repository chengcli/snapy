"""Validate and plot the completed GPU planar convection benchmark samples."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snapy-note-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from planar_compressible_reference import growth

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "planar_samples"
WINDOWS = ((0.4, 0.6), (0.6, 0.8), (0.8, 1.0))


def scientific(value):
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def read_run(name, directory=SAMPLES):
    path = directory / (name + ".csv")
    data = np.genfromtxt(path, delimiter=",", names=True)
    meta = json.loads(Path(str(path) + ".json").read_text())
    assert meta["device"].startswith("cuda:"), name
    assert data["diffusive_time"][-1] >= meta["end_diffusive_time"] - 1.0e-12
    assert np.all(np.isfinite(data.view(float))) and np.all(np.diff(data["time"]) > 0)
    return data, meta


def fit(data, lo, hi):
    select = (data["diffusive_time"] >= lo) & (data["diffusive_time"] <= hi)
    assert select.sum() >= 10
    t, amp = data["diffusive_time"][select], data["amplitude"][select]
    assert np.all(amp > 0)
    rate, intercept = np.polyfit(t, np.log(amp), 1)
    residual = np.log(amp) - rate * t - intercept
    return dict(
        window=[lo, hi], growth=float(rate), rms_log_residual=float(np.sqrt(np.mean(residual**2)))
    )


def neutral_root(ras, rates):
    # Three nearby runs resolve the weak curvature in sigma(Ra).
    roots = np.roots(np.polyfit(ras, rates, 2))
    roots = [float(r.real) for r in roots if abs(r.imag) < 1.0e-9 and min(ras) < r.real < max(ras)]
    assert len(roots) == 1
    return roots[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SAMPLES)
    args = parser.parse_args()
    reference = json.loads((HERE / "planar_reference_results.json").read_text())
    case = next(c for c in reference["cases"] if c["name"] == "stratified")
    k, exact = case["levels"][-1]["k"], case["levels"][-1]["Ra"]
    curves, grids, rows = {}, [], []
    ras = [680, 695, 710]
    predicted = [growth(48, k, case["a"], case["D"], ra)["real"] for ra in ras]
    for n in (16, 32, 64):
        group = []
        for label, ra, theory in zip(("low", "middle", "high"), ras, predicted):
            data, meta = read_run(f"n{n}-{label}", args.input_dir)
            assert meta["Ra"] == ra and meta["nz"] == n
            assert data["diffusive_time"][-1] >= 1.0 - 1.0e-12
            assert meta["a"] == case["a"] and meta["D"] == case["D"]
            assert abs(meta["gamma"] - case["gamma"]) < 1.0e-14 and meta["Pr"] == 1
            assert meta["seed"] == 1.0e-6 and meta["cfl"] == 0.8
            assert abs(meta["k"] - k) < 1.0e-12 and meta["retries"] == 0
            fits = [fit(data, *window) for window in WINDOWS]
            row = dict(
                name=f"n{n}-{label}",
                **meta,
                fits=fits,
                spectral_growth=theory,
                growth_error=fits[-1]["growth"] - theory,
            )
            rows.append(row)
            group.append(row)
            curves[row["name"]] = data
        roots = [
            neutral_root(ras, [r["fits"][j]["growth"] for r in group]) for j in range(len(WINDOWS))
        ]
        assert group[0]["fits"][-1]["growth"] < 0 < group[-1]["fits"][-1]["growth"]
        grids.append(
            dict(
                nz=n,
                nx=2 * n,
                neutral_by_window=roots,
                error=roots[-1] - exact,
                relative_error=(roots[-1] - exact) / exact,
            )
        )
    errors = np.array([g["error"] for g in grids])
    assert np.all(np.abs(errors[1:]) < np.abs(errors[:-1]))
    assert abs(grids[-1]["relative_error"]) < 0.001  # finest-grid target: 0.1%
    inverse_n2 = 1.0 / np.array([g["nz"] for g in grids]) ** 2
    extrapolation = np.polyfit(inverse_n2, [g["neutral_by_window"][-1] for g in grids], 1)
    continuum = float(extrapolation[1])
    observed_order = float(
        np.log(
            abs(
                (grids[0]["neutral_by_window"][-1] - grids[1]["neutral_by_window"][-1])
                / (grids[1]["neutral_by_window"][-1] - grids[2]["neutral_by_window"][-1])
            )
        )
        / np.log(2)
    )

    rest, rest_meta = read_run("rest", args.input_dir)
    assert rest["diffusive_time"][-1] >= 0.3 - 1.0e-12
    rest_result = dict(
        max_mode_amplitude=float(np.max(np.abs(rest["amplitude"]))),
        max_vertical_rms=float(rest["vertical_rms"].max()),
        max_temperature_drift=float(rest["temperature_drift"].max()),
        max_density_drift=float(rest["density_drift"].max()),
        **rest_meta,
    )
    assert rest_result["max_mode_amplitude"] < 1.0e-12
    assert rest_result["max_vertical_rms"] < 1.0e-8
    base = curves["n32-high"]
    controls = []
    for name in ("small-seed", "half-dt"):
        data, meta = read_run(name, args.input_dir)
        assert data["diffusive_time"][-1] >= 1.0 - 1.0e-12
        select = data["diffusive_time"] >= 0.4
        reference_amp = np.interp(
            data["diffusive_time"][select], base["diffusive_time"], base["amplitude"]
        )
        ratio = 1.0e-6 / meta["seed"] * data["amplitude"][select] / reference_amp
        scaling_error = float(np.max(np.abs(ratio - 1)))
        fits = [fit(data, *window) for window in WINDOWS]
        controls.append(
            dict(name=name, **meta, fits=fits, max_relative_amplitude_difference=scaling_error)
        )
        assert scaling_error < 0.002
    result = dict(
        reference=case,
        fits=rows,
        grids=grids,
        continuum_extrapolation=continuum,
        observed_order=observed_order,
        continuum_relative_error=(continuum - exact) / exact,
        rest=rest_result,
        controls=controls,
        note="One periodic fundamental wavelength; stationary onset. Fits are estimates, not formal confidence intervals.",
    )
    (HERE / "planar_onset_results.json").write_text(json.dumps(result, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for n in (16, 32, 64):
        r = [row for row in rows if row["nz"] == n]
        axes[0].plot(ras, [x["fits"][-1]["growth"] for x in r], "o-", label=f"GPU {n} × {2 * n}")
    axes[0].plot(ras, predicted, "k--", label="Compressible spectral")
    axes[0].axhline(0, color="gray", lw=0.7)
    axes[0].set(xlabel=r"$Ra_{sa}$", ylabel=r"Growth rate ($\chi_m/d^2$)")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        inverse_n2, [g["neutral_by_window"][-1] for g in grids], "o", label="GPU neutral fit"
    )
    xx = np.linspace(0, inverse_n2[0], 50)
    axes[1].plot(xx, np.polyval(extrapolation, xx), "--", label=r"Fit in $N_z^{-2}$")
    axes[1].axhline(exact, color="k", lw=0.8, label=f"Spectral: {exact:.5f}")
    axes[1].set(xlabel=r"$N_z^{-2}$", ylabel=r"Estimated $Ra_{sa,c}$")
    axes[1].legend(fontsize=8)
    for label, ra in zip(("low", "middle", "high"), ras):
        data = curves[f"n32-{label}"]
        use = data["diffusive_time"] >= 0.4
        norm = np.interp(0.4, data["diffusive_time"], data["amplitude"])
        axes[2].plot(data["diffusive_time"][use], data["amplitude"][use] / norm, label=f"Ra = {ra}")
    axes[2].set(xlabel=r"Time ($d^2/\chi_m$)", ylabel="Mode amplitude / amplitude at t = 0.4")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "figures/planar_compressible_onset.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    table = [
        ".. Generated by analyze_planar_onset.py.\n",
        ".. list-table:: GPU planar onset, a = 1, D = 0.5, gamma = 5/3, Pr = 1",
        "   :header-rows: 1\n",
        "   * - Vertical × horizontal cells",
        "     - Growth at Ra = 680",
        "     - Growth at Ra = 695",
        "     - Growth at Ra = 710",
        "     - Neutral Ra",
        "     - Relative error",
    ]
    for grid in grids:
        group = [row for row in rows if row["nz"] == grid["nz"]]
        table += [f"   * - {grid['nz']} × {grid['nx']}"]
        table += [f"     - {row['fits'][-1]['growth']:.7f}" for row in group]
        table += [
            f"     - {grid['neutral_by_window'][-1]:.6f}",
            f"     - {100 * grid['relative_error']:.5f}%",
        ]
    table += [
        "   * - Spectral reference",
        *[f"     - {v:.7f}" for v in predicted],
        f"     - {exact:.6f}",
        "     - --",
    ]
    (HERE / "planar_onset_tables.inc").write_text("\n".join(table) + "\n")
    finest = grids[-1]
    small = next(c for c in controls if c["name"] == "small-seed")
    half = next(c for c in controls if c["name"] == "half-dt")
    baseline_rate = next(r for r in rows if r["name"] == "n32-high")["fits"][-1]["growth"]
    window_shift = abs(finest["neutral_by_window"][-1] - finest["neutral_by_window"][-2])
    summary = (
        ".. Generated by analyze_planar_onset.py.\n\n"
        f"The finest GPU grid gives :math:`Ra_c={finest['neutral_by_window'][-1]:.6f}`, "
        f"a relative error of :math:`{100 * abs(finest['relative_error']):.5f}\\%` "
        f"against the compressible spectral result :math:`{exact:.6f}`. "
        f"The three-grid observed convergence order is :math:`{observed_order:.3f}`. "
        f"A linear extrapolation in :math:`N_z^{{-2}}` gives :math:`{continuum:.6f}` "
        f"(relative error :math:`{100 * abs((continuum - exact) / exact):.5f}\\%`). "
        "This is an empirical extrapolation, not a rigorous error bound.\n\n"
        f"Moving the fit window from 0.6--0.8 to 0.8--1.0 changes the finest-grid "
        f"neutral estimate by :math:`{window_shift:.6f}`. Halving the CFL factor "
        f"from 0.8 to 0.4 changes the growth rate at :math:`Ra=710`, "
        f":math:`N_z=32` by :math:`{scientific(abs(half['fits'][-1]['growth'] - baseline_rate))}` "
        f"and the normalized mode amplitude by at most "
        f":math:`{scientific(half['max_relative_amplitude_difference'])}`. Reducing the "
        f"seed from :math:`10^{{-6}}` to :math:`10^{{-7}}` changes the normalized "
        f"amplitude by at most :math:`{scientific(small['max_relative_amplitude_difference'])}`.\n\n"
        f"The unperturbed control, evolved to diffusive time 0.3, has maximum "
        f"vertical RMS velocity :math:`{scientific(rest_result['max_vertical_rms'])}` "
        f"and maximum temperature drift :math:`{scientific(rest_result['max_temperature_drift'])}`. "
        "No step retries occurred in the onset scan.\n"
    )
    (HERE / "planar_onset_summary.inc").write_text(summary)
    print(
        json.dumps(
            dict(
                grids=grids,
                continuum=continuum,
                order=observed_order,
                rest=rest_result,
                controls=controls,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
