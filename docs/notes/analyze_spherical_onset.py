"""Analyze completed six-panel GPU convection runs without assuming filename tags.

Strict mode requires bracketed neutral estimates on two refined grids at every
requested radius ratio. --allow-partial writes explicitly incomplete diagnostics.
CSV files without their completion metadata are reported and skipped. No GPU
work is launched. Example:

    OPENBLAS_NUM_THREADS=1 python docs/notes/analyze_spherical_onset.py --allow-partial

Use --output-dir /tmp/spherical-analysis to keep exploratory artifacts separate
from the note. --windows .05:.1,.1:.15,.15:.2 fixes fit intervals explicitly;
otherwise three successive quarters of each group's common duration are used.
"""

import argparse
from collections import defaultdict
from itertools import combinations
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snapy-note-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
GROUP_KEYS = (
    "eta",
    "ell",
    "N",
    "nr",
    "a",
    "D",
    "gamma",
    "Pr",
    "seed",
    "cfl",
    "diffusion_step_scale",
    "orientation",
    "eigenmode_seed",
    "dtype",
)
LIMITATIONS = [
    "Neutral estimates are zeros of fitted finite-time modal growth, not formal confidence intervals.",
    "Two Rayleigh numbers use linear interpolation; three use a quadratic inside a measured sign bracket.",
    "Two grids establish a refinement comparison, not a demonstrated asymptotic convergence order.",
    "One seeded harmonic/orientation does not establish the minimum over all discrete modes or orientations.",
    "Short-time fits can retain startup, acoustic, hydrostatic and seam errors; successive windows expose some of these effects.",
    "Metadata eigenmode_seed records use of a profile, not its physical correctness or reference-code version.",
    "Controls are reported numerically; no universal acceptable error or equilibrium-drift tolerance is assumed.",
    "diffusion_step_scale multiplies the conservative diffusion bound only; the acoustic cap still applies, so scale ratios are not actual timestep ratios.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(meta, fields=GROUP_KEYS):
    return tuple(meta[k] for k in fields)


def windows_for(end, explicit=None):
    # Final RK steps can leave roundoff-sized duration differences across grids.
    end = round(end, 12)
    windows = explicit or [(end / 4, end / 2), (end / 2, 3 * end / 4), (3 * end / 4, end)]
    require(
        all(0 <= lo < hi <= end + 1e-10 for lo, hi in windows),
        f"Fit windows extend beyond completed duration {end:g}",
    )
    return windows


def fit(data, window, min_points=10):
    lo, hi = window
    select = (data["diffusive_time"] >= lo - 1e-12) & (data["diffusive_time"] <= hi + 1e-12)
    t, amplitude = data["diffusive_time"][select], data["amplitude"][select]
    require(
        len(t) >= min_points, f"Only {len(t)} samples in window {lo:g}:{hi:g}; need {min_points}"
    )
    require(
        np.all(amplitude != 0) and np.all(np.sign(amplitude) == np.sign(amplitude[0])),
        "Mode changes sign or vanishes inside fit window",
    )
    log_amp = np.log(np.abs(amplitude))
    rate, intercept = np.polyfit(t, log_amp, 1)
    residual = log_amp - rate * t - intercept
    return dict(
        window=[float(lo), float(hi)],
        sample_count=len(t),
        sampled_interval=[float(t[0]), float(t[-1])],
        growth=float(rate),
        rms_log_residual=float(np.sqrt(np.mean(residual**2))),
        max_log_residual=float(np.max(np.abs(residual))),
    )


def read_run(path):
    meta = json.loads(Path(str(path) + ".json").read_text())
    # Runs before the precision option were unconditionally double precision.
    meta.setdefault("dtype", "float64")
    require(meta["dtype"] in ("float64", "float32"), "Unknown precision")
    needed = (*GROUP_KEYS, "Ra", "device", "end_diffusive_time", "cycles", "seconds", "retries")
    require(all(k in meta for k in needed), f"Missing metadata: {set(needed) - meta.keys()}")
    require(str(meta["device"]).startswith("cuda:"), "Run did not use a CUDA device")
    require(meta["retries"] == 0, "Step retries occurred")
    for name, value in meta.items():
        if isinstance(value, (int, float)):
            require(np.isfinite(value), f"Nonfinite metadata {name}")
    for name, target in (("a", 1), ("D", 0.5), ("gamma", 5 / 3), ("Pr", 1)):
        require(np.isclose(meta[name], target, rtol=1e-12, atol=0), f"Unexpected {name}")
    for name in ("N", "nr", "ell", "cycles"):
        require(meta[name] > 0 and int(meta[name]) == meta[name], f"Invalid {name}")
    require(
        0 < meta["eta"] < 1 and meta["Ra"] > 0 and meta["seed"] >= 0, "Invalid physical parameters"
    )
    require(
        all(meta[k] > 0 for k in ("cfl", "diffusion_step_scale", "end_diffusive_time", "seconds")),
        "Invalid run duration or timestep parameter",
    )
    require(isinstance(meta["eigenmode_seed"], bool), "eigenmode_seed must be boolean")
    require(meta["orientation"] in ("z", "diagonal"), "Unknown orientation")
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    columns = (
        "time",
        "diffusive_time",
        "amplitude",
        "radial_rms",
        "temperature_drift",
        "density_drift",
    )
    require(data.dtype.names and all(k in data.dtype.names for k in columns), "Missing CSV columns")
    require(len(data) >= 2, "Insufficient CSV rows")
    require(all(np.all(np.isfinite(data[k])) for k in data.dtype.names), "Nonfinite CSV values")
    # Older CSV output used 16 digits and could print a tiny final correction
    # at the same time as the preceding row. Only coalesce an indistinguishable
    # final pair; never hide a reversal or a duplicate inside the trajectory.
    eps = np.finfo(float).eps
    if len(data) > 2 and data["diffusive_time"][-1] <= data["diffusive_time"][-2]:
        same_endpoint = all(
            abs(data[name][-1] - data[name][-2]) <= 16 * eps * max(1.0, abs(data[name][-1]))
            for name in ("time", "diffusive_time")
        )
        same_state = all(
            np.isclose(data[name][-1], data[name][-2], rtol=1.0e-10, atol=1.0e-14)
            for name in columns[2:]
        )
        if same_endpoint and same_state:
            data = np.concatenate((data[:-2], data[-1:]))
            meta["roundoff_endpoint_rows_removed"] = 1
    for name in ("time", "diffusive_time"):
        require(data[name][0] >= 0 and np.all(np.diff(data[name]) > 0), f"Nonmonotone {name}")
    require(
        np.isclose(data["diffusive_time"][-1], meta["end_diffusive_time"], rtol=1e-9, atol=1e-11),
        "CSV final time does not match completion metadata",
    )
    if "chi_mid" in meta:
        require(
            np.allclose(
                data["diffusive_time"], data["time"] * meta["chi_mid"], rtol=1e-10, atol=1e-12
            ),
            "Physical and diffusive times disagree",
        )
    return data, meta


def neutral_root(ras, rates):
    """Use a unique adjacent sign bracket; never extrapolate or combine repeats."""
    order = np.argsort(ras)
    ras, rates = np.asarray(ras)[order], np.asarray(rates)[order]
    require(len(ras) >= 2 and np.all(np.diff(ras) > 0), "Need distinct Rayleigh numbers")
    require(np.all(np.isfinite(rates)), "Nonfinite growth rates")
    require(np.all(np.diff(rates) > 0), "Growth is not monotone with Rayleigh number")
    require(rates[0] < 0 < rates[-1], "No measured decay/growth bracket")
    crossings = np.flatnonzero(rates[:-1] * rates[1:] < 0)
    zeros = np.flatnonzero(rates == 0)
    if len(zeros):
        i = int(zeros[0])
        bracket = [float(ras[i - 1]), float(ras[i + 1])]
        return dict(
            Ra=float(ras[i]),
            method="measured zero with two-sided sign bracket",
            bracket=bracket,
            used_Ra=[float(x) for x in ras[i - 1 : i + 2]],
        )
    require(len(crossings) == 1, "Ambiguous sign bracket")
    i = int(crossings[0])
    ids = [i, i + 1]
    if len(ras) > 2:
        extra = [j for j in (i - 1, i + 2) if 0 <= j < len(ras)]
        ids.append(min(extra, key=lambda j: abs(ras[j] - np.mean(ras[i : i + 2]))))
    ids.sort()
    # Centering avoids conditioning loss for nearly coincident large Ra values.
    center, scale = np.mean(ras[ids]), np.ptp(ras[ids])
    coefficients = np.polyfit((ras[ids] - center) / scale, rates[ids], len(ids) - 1)
    roots = np.roots(coefficients)
    valid = [
        float(r.real * scale + center)
        for r in roots
        if abs(r.imag) < 1e-9 and ras[i] <= r.real * scale + center <= ras[i + 1]
    ]
    require(len(valid) == 1, "Polynomial has no unique root within the measured bracket")
    return dict(
        Ra=valid[0],
        method="quadratic (three nearest points)" if len(ids) == 3 else "linear (two points)",
        bracket=[float(ras[i]), float(ras[i + 1])],
        used_Ra=[float(x) for x in ras[ids]],
    )


def reference_for(reference, meta):
    case = next(
        (c for c in reference["cases"] if np.isclose(c["eta"], meta["eta"], rtol=0, atol=1e-12)),
        None,
    )
    require(case is not None, f"No reference at eta={meta['eta']}")
    if case["ell"] == meta["ell"]:
        modal_ra = case["Ra"]
    else:
        from spherical_compressible_reference import neutral

        modal_ra = neutral(48, meta["eta"], meta["ell"])
        if isinstance(modal_ra, dict):
            modal_ra = modal_ra["Ra"]
    return dict(
        modal_Ra=float(modal_ra),
        critical_Ra=case["Ra"],
        critical_ell=case["ell"],
        planar_Ra=reference["planar"]["Ra"],
    )


def analyze(directory, reference, explicit_windows=None, min_points=10, exclude=()):
    runs, data_by_name, invalid, incomplete, excluded = [], {}, [], [], []
    for path in sorted(directory.glob("*.csv")):
        if any(path.match(pattern) for pattern in exclude):
            excluded.append(path.name)
            continue
        if not Path(str(path) + ".json").exists():
            incomplete.append(path.name)
            continue
        try:
            data, meta = read_run(path)
            row = dict(name=path.name, **meta)
            row["max_radial_rms"] = float(np.max(np.abs(data["radial_rms"])))
            row["max_temperature_drift"] = float(np.max(np.abs(data["temperature_drift"])))
            row["max_density_drift"] = float(np.max(np.abs(data["density_drift"])))
            row["max_mode_amplitude"] = float(np.max(np.abs(data["amplitude"])))
            runs.append(row)
            data_by_name[path.name] = data
        except (ValueError, KeyError, TypeError, OSError) as error:
            invalid.append(dict(name=path.name, reason=str(error)))
    groups = defaultdict(list)
    for row in runs:
        if row["seed"] > 0:
            groups[key(row)].append(row)
    grids = []
    for members in groups.values():
        parameters = {k: members[0][k] for k in GROUP_KEYS}
        group = dict(
            parameters=parameters,
            runs=[r["name"] for r in members],
            neutral_by_window=[],
            issues=[],
        )
        try:
            group["reference"] = reference_for(reference, parameters)
            windows = windows_for(min(r["end_diffusive_time"] for r in members), explicit_windows)
            group["windows"] = windows
            for row in members:
                row["fits"] = [fit(data_by_name[row["name"]], w, min_points) for w in windows]
                row["growth_window_range"] = float(np.ptp([f["growth"] for f in row["fits"]]))
                if len(row["fits"]) >= 2:
                    row["last_window_growth_shift"] = (
                        row["fits"][-1]["growth"] - row["fits"][-2]["growth"]
                    )
            for j, window in enumerate(windows):
                try:
                    root = neutral_root(
                        [r["Ra"] for r in members], [r["fits"][j]["growth"] for r in members]
                    )
                    root["window"] = list(window)
                    root["relative_error"] = root["Ra"] / group["reference"]["modal_Ra"] - 1
                    group["neutral_by_window"].append(root)
                except ValueError as error:
                    group["neutral_by_window"].append(None)
                    group["issues"].append(f"Window {window}: {error}")
            if group["neutral_by_window"][-1] is not None:
                group["neutral"] = group["neutral_by_window"][-1]
            if len(group["neutral_by_window"]) > 1 and all(group["neutral_by_window"][-2:]):
                group["last_window_Ra_shift"] = (
                    group["neutral_by_window"][-1]["Ra"] - group["neutral_by_window"][-2]["Ra"]
                )
        except (ValueError, KeyError, TypeError) as error:
            group["issues"].append(str(error))
        grids.append(group)

    controls = []
    nonzero = [r for r in runs if r["seed"] > 0 and "fits" in r]
    for left, right in combinations(nonzero, 2):
        varying = [k for k in GROUP_KEYS if left[k] != right[k]]
        if (
            left["Ra"] != right["Ra"]
            or not varying
            or not set(varying) <= {"seed", "cfl", "diffusion_step_scale", "orientation", "dtype"}
        ):
            continue
        # A control varies either seed, orientation, or timestep, never several.
        categories = set(
            "timestep" if k in ("cfl", "diffusion_step_scale") else
            "precision" if k == "dtype" else k for k in varying)
        if len(categories) != 1:
            continue
        common_end = min(left["end_diffusive_time"], right["end_diffusive_time"])
        try:
            window = windows_for(common_end, explicit_windows)[-1]
            ld, rd = data_by_name[left["name"]], data_by_name[right["name"]]
            lf, rf = fit(ld, window, min_points), fit(rd, window, min_points)
            use = (rd["diffusive_time"] >= window[0]) & (rd["diffusive_time"] <= window[1])
            denominator = (
                np.interp(rd["diffusive_time"][use], ld["diffusive_time"], ld["amplitude"])
                / left["seed"]
            )
            require(np.all(denominator != 0), "Zero control amplitude")
            difference = rd["amplitude"][use] / right["seed"] / denominator - 1
            controls.append(
                dict(
                    kind=next(iter(categories)),
                    runs=[left["name"], right["name"]],
                    varied_parameters=varying,
                    window=list(window),
                    growth_difference=rf["growth"] - lf["growth"],
                    max_seed_normalized_amplitude_difference=float(np.max(np.abs(difference))),
                )
            )
        except ValueError as error:
            controls.append(dict(runs=[left["name"], right["name"]], issue=str(error)))
    return dict(
        runs=runs,
        groups=grids,
        controls=controls,
        rest_controls=[r for r in runs if r["seed"] == 0],
        invalid_runs=invalid,
        incomplete_runs=incomplete,
        excluded_runs=excluded,
    )


def coverage(result, required, min_grids, seed, cfl, orientation, step_scale):
    eligible = [
        g
        for g in result["groups"]
        if "neutral" in g
        and g["parameters"]["seed"] == seed
        and g["parameters"]["cfl"] == cfl
        and g["parameters"]["orientation"] == orientation
        and g["parameters"]["eigenmode_seed"]
        and g["parameters"]["ell"] == g["reference"]["critical_ell"]
    ]
    if step_scale is None and eligible:
        counts = defaultdict(set)
        for g in eligible:
            p = g["parameters"]
            counts[p["diffusion_step_scale"]].add((p["eta"], p["N"], p["nr"]))
        step_scale = max(sorted(counts), key=lambda s: len(counts[s]))
    selected = [g for g in eligible if g["parameters"]["diffusion_step_scale"] == step_scale]
    ratios, missing = [], []
    for eta in required:
        grids = sorted(
            {
                (g["parameters"]["nr"], g["parameters"]["N"])
                for g in selected
                if abs(g["parameters"]["eta"] - eta) < 1e-12
            }
        )
        # Longest chain refining both coordinates, with at least one strict increase.
        lengths = [1] * len(grids)
        for j, (nr, n) in enumerate(grids):
            for i, (previous_nr, previous_n) in enumerate(grids[:j]):
                if previous_nr <= nr and previous_n <= n:
                    lengths[j] = max(lengths[j], lengths[i] + 1)
        sufficient = max(lengths, default=0) >= min_grids
        ratios.append(
            dict(eta=eta, grids=[dict(nr=r, N=n) for r, n in grids], sufficient=sufficient)
        )
        if not sufficient:
            missing.append(eta)
    return dict(
        required_ratios=list(required),
        minimum_refined_grids=min_grids,
        baseline=dict(seed=seed, cfl=cfl, orientation=orientation, diffusion_step_scale=step_scale),
        ratios=ratios,
        missing_or_unrefined_ratios=missing,
        complete=not missing and not result["invalid_runs"],
    )


def refinement_comparisons(groups):
    """Compare matched grids; Richardson extrapolations state their assumptions."""
    families = defaultdict(list)
    fields = tuple(k for k in GROUP_KEYS if k not in ("N", "nr"))
    for group in groups:
        if "neutral" in group:
            families[key(group["parameters"], fields)].append(group)
    comparisons = []
    for members in families.values():
        members.sort(key=lambda g: (g["parameters"]["nr"], g["parameters"]["N"]))
        for coarse, fine in zip(members, members[1:]):
            cp, fp = coarse["parameters"], fine["parameters"]
            radial, angular = fp["nr"] / cp["nr"], fp["N"] / cp["N"]
            if radial < 1 or angular < 1:
                continue
            comparison = dict(
                parameters={k: cp[k] for k in fields},
                coarse_grid={"nr": cp["nr"], "N": cp["N"]},
                fine_grid={"nr": fp["nr"], "N": fp["N"]},
                radial_refinement_factor=radial,
                angular_refinement_factor=angular,
                same_fit_window=bool(
                    np.allclose(
                        coarse["neutral"]["window"], fine["neutral"]["window"], rtol=0, atol=1e-10
                    )
                ),
                neutral_change=fine["neutral"]["Ra"] - coarse["neutral"]["Ra"],
                absolute_error_decreased=abs(fine["neutral"]["relative_error"])
                < abs(coarse["neutral"]["relative_error"]),
            )
            if (
                radial > 1
                and np.isclose(radial, angular, rtol=1e-12)
                and comparison["same_fit_window"]
            ):
                comparison["assumed_second_order_extrapolation"] = (
                    radial**2 * fine["neutral"]["Ra"] - coarse["neutral"]["Ra"]
                ) / (radial**2 - 1)
                comparison["extrapolation_note"] = (
                    "Assumes second-order error under simultaneous radial/angular refinement; not a measured order or rigorous bound."
                )
            comparisons.append(comparison)
        # With three proportional grids, solve the unequal-spacing error-ratio equation.
        if len(members) >= 3:
            coarse, middle, fine = members[-3:]
            n = np.array([g["parameters"]["nr"] for g in (coarse, middle, fine)], dtype=float)
            horizontal = np.array([g["parameters"]["N"] for g in (coarse, middle, fine)])
            values = np.array([g["neutral"]["Ra"] for g in (coarse, middle, fine)])
            if (
                np.all(np.diff(n) > 0)
                and np.allclose(horizontal / n, horizontal[0] / n[0], rtol=1e-12)
                and all(
                    np.allclose(
                        g["neutral"]["window"], fine["neutral"]["window"], rtol=0, atol=1e-10
                    )
                    for g in (coarse, middle)
                )
            ):
                delta = np.diff(values)
                if delta[0] * delta[1] > 0:
                    from scipy.optimize import brentq

                    def equation(order):
                        powers = (n / n[0]) ** (-order)
                        return (powers[0] - powers[1]) / (powers[1] - powers[2]) - delta[0] / delta[
                            1
                        ]

                    try:
                        order = float(brentq(equation, 0.05, 8))
                        ratio = n[2] / n[1]
                        comparisons.append(
                            dict(
                                parameters={k: fine["parameters"][k] for k in fields},
                                grids=[
                                    {"nr": g["parameters"]["nr"], "N": g["parameters"]["N"]}
                                    for g in (coarse, middle, fine)
                                ],
                                observed_order=order,
                                empirical_extrapolation=float(
                                    (ratio**order * values[2] - values[1]) / (ratio**order - 1)
                                ),
                                extrapolation_note="Three-grid empirical estimate; finite-time fit errors and higher-order terms remain.",
                            )
                        )
                    except ValueError:
                        pass
    return comparisons


def write_artifacts(result, reference, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "spherical_onset_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    measured = [g for g in result["groups"] if "neutral" in g]
    table = [
        ".. Generated by analyze_spherical_onset.py.\n",
        ".. list-table:: Measured modal neutral points (only completed brackets)",
        "   :header-rows: 1\n",
        "   * - Radius ratio",
        "     - Degree",
        "     - nr / N per panel",
        "     - Seed / CFL / diffusion step scale / axis",
        "     - Neutral Ra",
        "     - Same-degree spectral Ra",
        "     - Relative error",
        "     - Method / final window",
    ]
    for g in measured:
        p, root = g["parameters"], g["neutral"]
        lo, hi = root["window"]
        table += [
            f"   * - {p['eta']:g}",
            f"     - {p['ell']}",
            f"     - {p['nr']} / {p['N']}",
            f"     - {p['seed']:g} / {p['cfl']:g} / {p['diffusion_step_scale']:g} / {p['orientation']}",
            f"     - {root['Ra']:.6f}",
            f"     - {g['reference']['modal_Ra']:.6f}",
            f"     - {100 * root['relative_error']:+.4f}%",
            f"     - {root['method']}; {lo:g}--{hi:g}",
        ]
    if not measured:
        table = [
            ".. Generated by analyze_spherical_onset.py.\n",
            "No completed sign-changing neutral bracket is available.\n",
        ]
    (output / "spherical_onset_tables.inc").write_text("\n".join(table) + "\n")
    status = (
        "Coverage requirement met"
        if result["coverage"]["complete"]
        else "PARTIAL DIAGNOSTIC: coverage requirement not met"
    )
    summary = [
        ".. Generated by analyze_spherical_onset.py.\n",
        f"**{status}.**\n",
        f"Analyzed {len(result['runs'])} completed GPU runs, totaling {result['total_cycles']:,} steps "
        f"and {result['total_gpu_run_seconds']:.1f} summed run seconds (not elapsed batch time). "
        f"There are {len(measured)} bracketed modal neutral estimates; "
        f"{len(result['incomplete_runs'])} running/incomplete CSV files were skipped and "
        f"{len(result['invalid_runs'])} completed runs failed input validation.\n",
        "The required baseline is ``"
        + json.dumps(result["coverage"]["baseline"], sort_keys=True)
        + "``.\n",
    ]
    missing = result["coverage"]["missing_or_unrefined_ratios"]
    if missing:
        summary += [
            "Missing or insufficiently refined radius ratios: "
            + ", ".join(f"{e:g}" for e in missing)
            + ".\n"
        ]
    summary += [
        "The plotted spectral curve includes radius ratio 0.9999 when present in the reference file; "
        "it is an independent continuum calculation, not a GPU measurement.\n",
        f"Available matched controls: {sum('growth_difference' in c for c in result['controls'])}; "
        f"zero-seed controls: {len(result['rest_controls'])}. "
        "Missing controls have not been marked as passed.\n",
    ]
    shifts = [abs(g["last_window_Ra_shift"]) for g in measured if "last_window_Ra_shift" in g]
    if shifts:
        summary.append(
            f"Across {len(shifts)} brackets present in both final windows, the absolute neutral-Ra "
            f"shift ranges from {min(shifts):.6g} to {max(shifts):.6g}. "
            "These shifts measure window sensitivity, not statistical uncertainty.\n"
        )
    for kind in ("timestep", "seed", "orientation"):
        controls = [
            c for c in result["controls"] if c.get("kind") == kind and "growth_difference" in c
        ]
        if controls:
            summary.append(
                f"The {len(controls)} matched {kind} comparisons have maximum absolute "
                f"growth-rate change {max(abs(c['growth_difference']) for c in controls):.6g} "
                f"and maximum seed-normalized amplitude difference "
                f"{max(c['max_seed_normalized_amplitude_difference'] for c in controls):.6g}.\n"
            )
    if result["rest_controls"]:
        summary.append(
            f"Across zero-seed controls the largest radial RMS velocity is "
            f"{max(r['max_radial_rms'] for r in result['rest_controls']):.6g}, "
            f"and the largest temperature drift is "
            f"{max(r['max_temperature_drift'] for r in result['rest_controls']):.6g}.\n"
        )
    summary += [text + "\n" for text in LIMITATIONS]
    (output / "spherical_onset_summary.inc").write_text("\n".join(summary))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cases = sorted(reference["cases"], key=lambda c: c["eta"])
    axes[0].plot(
        [c["eta"] for c in cases], [c["Ra"] for c in cases], "k.-", label="Spherical spectral"
    )
    axes[0].axhline(reference["planar"]["Ra"], color="gray", ls="--", label="Planar spectral limit")
    families = defaultdict(list)
    for g in measured:
        p = g["parameters"]
        label = f"nr={p['nr']}, seed={p['seed']:g}, CFL={p['cfl']:g}, scale={p['diffusion_step_scale']:g}, {p['orientation']}"
        families[label].append(g)
    for label, groups in families.items():
        angular = sorted({g["parameters"]["N"] for g in groups})
        label += f", N={angular[0]}" + (f"--{angular[-1]}" if len(angular) > 1 else "")
        axes[0].scatter(
            [g["parameters"]["eta"] for g in groups],
            [g["neutral"]["Ra"] for g in groups],
            label=label,
        )
    axes[0].set(xlabel=r"Radius ratio $\eta=r_i/r_o$", ylabel="Critical / measured modal Ra")
    axes[0].legend(fontsize=7)
    # Show up to three well-populated groups across the available eta range.
    available = [
        g
        for g in result["groups"]
        if all("fits" in next(r for r in result["runs"] if r["name"] == name) for name in g["runs"])
    ]
    chosen = []
    etas = sorted({g["parameters"]["eta"] for g in available})
    for eta in [etas[i] for i in sorted(set([0, len(etas) // 2, len(etas) - 1]))] if etas else []:
        chosen.append(
            max(
                (g for g in available if g["parameters"]["eta"] == eta),
                key=lambda g: (len(g["runs"]), g["parameters"]["nr"], g["parameters"]["N"]),
            )
        )
    for g in chosen:
        rows = sorted([r for r in result["runs"] if r["name"] in g["runs"]], key=lambda r: r["Ra"])
        p = g["parameters"]
        axes[1].plot(
            [r["Ra"] for r in rows],
            [r["fits"][-1]["growth"] for r in rows],
            "o-",
            label=f"eta={p['eta']:g}, nr/N={p['nr']}/{p['N']}",
        )
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set(xlabel="Ra", ylabel=r"Final-window growth ($\chi_m/d^2$)")
    if chosen:
        axes[1].legend(fontsize=7)
    controls = [
        c for c in result["controls"] if c.get("kind") == "timestep" and "growth_difference" in c
    ]
    if controls:
        axes[2].bar(range(len(controls)), [abs(c["growth_difference"]) for c in controls])
        axes[2].set(
            xlabel="Matched timestep-control pair (JSON order)",
            ylabel="Absolute growth-rate change",
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "No completed matched\ntimestep control available",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_axis_off()
    fig.suptitle(status, fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "figures/spherical_compressible_onset.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HERE / "spherical_samples")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--reference", type=Path, default=HERE / "spherical_reference_results.json")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write explicitly incomplete diagnostics, never a completion claim",
    )
    parser.add_argument(
        "--require-coverage", default=",".join(map(str, ETAS)), help="Comma-separated radius ratios"
    )
    parser.add_argument(
        "--min-grids", type=int, default=2, help="Required comparable refined grids per ratio"
    )
    parser.add_argument("--windows", help="Comma-separated lo:hi intervals in diffusive time")
    parser.add_argument("--min-fit-points", type=int, default=10)
    parser.add_argument("--baseline-seed", type=float, default=1e-6)
    parser.add_argument("--baseline-cfl", type=float, default=0.8)
    parser.add_argument("--baseline-orientation", choices=("z", "diagonal"), default="z")
    parser.add_argument(
        "--baseline-step-scale", type=float, help="Default: single scale with most eligible grids"
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Explicit CSV filename glob to exclude; recorded in JSON",
    )
    args = parser.parse_args()
    require(args.min_grids >= 1 and args.min_fit_points >= 3, "Invalid grid/fit requirement")
    required = tuple(map(float, args.require_coverage.split(",")))
    require(required and all(0 < eta < 1 for eta in required), "Invalid coverage ratios")
    windows = (
        [tuple(map(float, item.split(":"))) for item in args.windows.split(",")]
        if args.windows
        else None
    )
    if windows:
        require(all(len(w) == 2 for w in windows), "Windows must be lo:hi")
    require(args.input_dir.is_dir(), f"Input directory does not exist: {args.input_dir}")
    reference = json.loads(args.reference.read_text())
    result = analyze(args.input_dir, reference, windows, args.min_fit_points, args.exclude)
    result["coverage"] = coverage(
        result,
        required,
        args.min_grids,
        args.baseline_seed,
        args.baseline_cfl,
        args.baseline_orientation,
        args.baseline_step_scale,
    )
    result.update(
        reference_file=str(args.reference.resolve()),
        input_directory=str(args.input_dir.resolve()),
        reference_curve=[{k: c[k] for k in ("eta", "ell", "Ra")} for c in reference["cases"]],
        planar_reference=reference["planar"],
        limitations=LIMITATIONS,
        total_cycles=sum(r["cycles"] for r in result["runs"]),
        total_gpu_run_seconds=sum(r["seconds"] for r in result["runs"]),
        window_policy="explicit" if windows else "last three quarters of common group duration",
    )
    result["status"] = (
        "coverage_complete" if result["coverage"]["complete"] else "partial_diagnostic"
    )
    result["refinement_comparisons"] = refinement_comparisons(result["groups"])
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "coverage", "invalid_runs", "incomplete_runs", "total_cycles")
            },
            indent=2,
        )
    )
    if not result["coverage"]["complete"] and not args.allow_partial:
        raise SystemExit(
            "Coverage/input validation failed; no artifacts written. Use --allow-partial for labeled diagnostics."
        )
    if not any("fits" in r for r in result["runs"]):
        raise SystemExit("No completed run has adequate fit data; no artifacts written.")
    write_artifacts(result, reference, args.output_dir)


if __name__ == "__main__":
    main()
