"""Supplemental spherical onset controls from completed raw CSV/metadata pairs.

No simulations are launched. The fixed study requires the eta=0.95, nr/N=12/132
float64 triplets ending at 0.2 and 0.6, plus eta=0.9, nr/N=12/72 controls at
degrees 20/21/22 about z and degree 21 about the diagonal. Selection uses metadata,
not filename tags. Each triplet must contain exactly one run at each of
0.94, 1, and 1.06 times its same-degree spectral Ra.

Example preview (partial results stay explicitly labeled):
    OPENBLAS_NUM_THREADS=1 python docs/notes/analyze_spherical_controls.py \
        --allow-partial --output-dir /tmp/spherical-controls

Outputs: spherical_control_study.inc, spherical_control_study.json, and
figures/spherical_control_study.png. The include contains the figure directive.
"""

import argparse
import json
from pathlib import Path

import analyze_spherical_onset as onset

np, plt = onset.np, onset.plt
HERE = Path(__file__).resolve().parent
FACTORS = (0.94, 1.0, 1.06)
SHORT_WINDOWS = ((0.05, 0.1), (0.1, 0.15), (0.15, 0.2))
LONG_WINDOWS = (*SHORT_WINDOWS, (0.3, 0.4), (0.45, 0.6))
COMMON = dict(
    a=1.0,
    D=0.5,
    gamma=5 / 3,
    Pr=1.0,
    seed=1e-6,
    cfl=0.8,
    diffusion_step_scale=8.0,
    eigenmode_seed=True,
    dtype="float64",
    nr=12,
)


def matches(meta, expected):
    return all(
        np.isclose(meta[name], value, rtol=1e-12, atol=0)
        if isinstance(value, (float, int)) and not isinstance(value, bool)
        else meta[name] == value
        for name, value in expected.items()
    )


def read_completed(directory):
    """Ignore trajectories until their completion sidecar is present."""
    runs, invalid, incomplete = [], [], []
    for path in sorted(directory.glob("*.csv")):
        if not Path(str(path) + ".json").is_file():
            incomplete.append(str(path.resolve()))
            continue
        try:
            data, meta = onset.read_run(path)
            runs.append(dict(path=str(path.resolve()), metadata=meta, data=data))
        except (ValueError, KeyError, TypeError, OSError) as error:
            invalid.append(dict(path=str(path.resolve()), reason=str(error)))
    return runs, dict(directory=str(directory.resolve()), invalid=invalid, incomplete=incomplete)


def triplet(runs, parameters, end, spectral_ra, windows, min_points):
    targets = [factor * spectral_ra for factor in FACTORS]
    eligible = [
        run
        for run in runs
        if matches(run["metadata"], parameters)
        and np.isclose(run["metadata"]["end_diffusive_time"], end, rtol=0, atol=1e-10)
    ]
    selected, issues = [], []
    for ra in targets:
        candidates = [
            r for r in eligible if np.isclose(r["metadata"]["Ra"], ra, rtol=1e-10, atol=0)
        ]
        if len(candidates) == 1:
            selected.append(candidates[0])
        else:
            issues.append(f"Expected one completed run at Ra={ra:.9g}; found {len(candidates)}.")
    report = dict(
        parameters=parameters,
        end_diffusive_time=end,
        spectral_Ra=spectral_ra,
        required_Ra=targets,
        matched_count=len(selected),
        issues=issues,
        candidate_runs=[dict(path=r["path"], metadata=r["metadata"]) for r in eligible],
        runs=[dict(path=r["path"], metadata=r["metadata"]) for r in selected],
        windows=[],
    )
    for window in windows:
        row = dict(window=list(window), fits=[], neutral=None, issues=[])
        # Never form a neutral estimate from a partial triplet or mix repeat runs.
        if issues:
            row["issues"].append("Matched completed triplet unavailable; no window estimate.")
        else:
            try:
                onset.windows_for(end, [window])
                for run in selected:
                    fitted = onset.fit(run["data"], window, min_points)
                    row["fits"].append(dict(path=run["path"], Ra=run["metadata"]["Ra"], **fitted))
                root = onset.neutral_root(
                    [f["Ra"] for f in row["fits"]], [f["growth"] for f in row["fits"]]
                )
                row["neutral"] = dict(**root, relative_error=root["Ra"] / spectral_ra - 1)
            except ValueError as error:
                row["issues"].append(str(error))
        row["status"] = "complete" if row["neutral"] is not None else "partial_diagnostic"
        report["windows"].append(row)
    report["status"] = (
        "complete"
        if all(w["status"] == "complete" for w in report["windows"])
        else "partial_diagnostic"
    )
    return report


def analyze(input_dir, reference, long_dir=None, min_points=10):
    onset.require(input_dir.is_dir(), f"Input directory does not exist: {input_dir}")
    onset.require(min_points >= 3, "Need at least three fit points")
    long_dir = long_dir if long_dir is not None else input_dir / "long_window"
    onset.require(
        long_dir.resolve() != input_dir.resolve(), "Long and short directories must differ"
    )
    short_runs, short_audit = read_completed(input_dir)
    long_runs, long_audit = read_completed(long_dir)
    parameters = dict(COMMON, eta=0.95, ell=43, N=132, orientation="z")
    spectral = onset.reference_for(reference, parameters)["modal_Ra"]
    short = triplet(short_runs, parameters, 0.2, spectral, SHORT_WINDOWS, min_points)
    long = triplet(long_runs, parameters, 0.6, spectral, LONG_WINDOWS, min_points)
    repeats = []
    for original, repeat in zip(short["windows"], long["windows"]):
        row = dict(window=original["window"], status="partial_diagnostic")
        if original["neutral"] is not None and repeat["neutral"] is not None:
            row.update(
                status="complete",
                short_Ra=original["neutral"]["Ra"],
                long_Ra=repeat["neutral"]["Ra"],
                neutral_difference=repeat["neutral"]["Ra"] - original["neutral"]["Ra"],
                growth_differences=[
                    dict(
                        Ra=long_fit["Ra"],
                        short_growth=short_fit["growth"],
                        long_growth=long_fit["growth"],
                        difference=long_fit["growth"] - short_fit["growth"],
                    )
                    for short_fit, long_fit in zip(original["fits"], repeat["fits"])
                ],
            )
        else:
            row["reason"] = "Both original and long-run window brackets are required."
        repeats.append(row)
    modes = []
    for ell, orientation in ((20, "z"), (21, "z"), (22, "z"), (21, "diagonal")):
        p = dict(COMMON, eta=0.9, ell=ell, N=72, orientation=orientation)
        modal_ra = onset.reference_for(reference, p)["modal_Ra"]
        modes.append(triplet(short_runs, p, 0.2, modal_ra, [SHORT_WINDOWS[-1]], min_points))
    baseline = modes[1]["windows"][0]["neutral"]
    for mode in modes:
        root = mode["windows"][0]["neutral"]
        if root is not None and baseline is not None:
            mode["neutral_difference_from_ell21_z"] = root["Ra"] - baseline["Ra"]
    complete = all(g["status"] == "complete" for g in (short, long, *modes))
    return dict(
        status="control_coverage_complete" if complete else "partial_diagnostic",
        scope=(
            "Fixed supplemental controls only; "
            "not the eight-ratio onset coverage or an accuracy certification."
        ),
        short=short,
        long=long,
        short_window_repeats=repeats,
        mode_controls=modes,
        input_audit=dict(short=short_audit, long=long_audit),
        fit_min_points=min_points,
        repeat_sampling_note=(
            "The original short run clips its final timestep to diffusive time 0.2 and "
            "writes that endpoint; the long run normally steps past 0.2 without clipping there. "
            "The 0.15--0.2 fits therefore need not use the same sample times or sample count. "
            "Even samples of an identical underlying trajectory can yield different fitted "
            "rates when growth varies within the window. A repeated-window difference can "
            "thus reflect endpoint sampling as well as changes in the numerical trajectory; "
            "it alone does not establish a physical growth-rate change. The fits use each "
            "run's original samples without resampling. Sample counts, sampled intervals, "
            "and source provenance are retained in JSON."
        ),
        limitations=[
            "All reported estimates require three completed float64 runs "
            "at matched physical and numerical parameters.",
            "Neutral estimates require a measured decay/growth bracket; "
            "missing brackets are not extrapolated.",
            "Window variation and repeated-window differences are diagnostics, "
            "not statistical uncertainty.",
            "The four sampled modes/orientations do not establish the global discrete minimum.",
            "Unrelated pending or invalid CSV files are audited "
            "but do not determine supplemental control coverage.",
        ],
    )


def write_artifacts(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "spherical_control_study.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    status = (
        "Supplemental control coverage complete"
        if result["status"] == "control_coverage_complete"
        else "PARTIAL DIAGNOSTIC: supplemental controls incomplete"
    )
    short, long = result["short"], result["long"]
    lines = [
        ".. Generated by analyze_spherical_controls.py.\n",
        f"**{status}.**\n",
        "This status covers only the requested supplemental controls. It does not certify "
        "asymptotic convergence or completion of the eight-ratio onset study.\n",
        "The duration study uses eta=0.95, degree 43, nr/N=12/132, seed=1e-6, "
        "CFL=0.8, diffusion step scale 8, float64, and the z orientation. "
        f"The same-degree spectral reference is Ra={long['spectral_Ra']:.6f}. "
        f"Matched completed runs: original duration 0.2, {short['matched_count']}/3; "
        f"long duration 0.6, {long['matched_count']}/3.\n",
        "Every long-run window uses the same matched triplet. The first three windows "
        "are compared with independently fitted original 0.2 runs from the sample root. "
        "Differences below are long minus original.\n",
        result["repeat_sampling_note"] + "\n",
        ".. list-table:: Duration and repeated-window neutral estimates",
        "   :header-rows: 1\n",
        "   * - Diffusive-time window",
        "     - Long-run neutral Ra",
        "     - Original neutral Ra",
        "     - Repeat difference",
        "     - Long-window status",
    ]

    def number(root):
        return f"{root['Ra']:.6f}" if root is not None else "unavailable"

    for i, row in enumerate(long["windows"]):
        original = short["windows"][i]["neutral"] if i < len(short["windows"]) else None
        repeat = result["short_window_repeats"][i] if i < len(short["windows"]) else {}
        difference = repeat.get("neutral_difference")
        lo, hi = row["window"]
        lines += [
            f"   * - {lo:g}--{hi:g}",
            f"     - {number(row['neutral'])}",
            f"     - {number(original) if i < len(short['windows']) else 'not sampled'}",
            f"     - {difference:+.6g}" if difference is not None else "     - unavailable",
            f"     - {row['status']}",
        ]
    lines += [
        "\n.. list-table:: Sampled mode and orientation thresholds at eta=0.9, nr/N=12/72",
        "   :header-rows: 1\n",
        "   * - Degree / axis",
        "     - Completed matches",
        "     - Neutral Ra",
        "     - Same-degree spectral Ra",
        "     - Difference from degree 21 z",
        "     - Status",
    ]
    for mode in result["mode_controls"]:
        p = mode["parameters"]
        difference = mode.get("neutral_difference_from_ell21_z")
        lines += [
            f"   * - {p['ell']} / {p['orientation']}",
            f"     - {mode['matched_count']}/3",
            f"     - {number(mode['windows'][0]['neutral'])}",
            f"     - {mode['spectral_Ra']:.6f}",
            f"     - {difference:+.6g}" if difference is not None else "     - unavailable",
            f"     - {mode['status']}",
        ]
    lines.append(
        "\nAll mode/orientation estimates use window 0.15--0.2, float64, seed=1e-6, "
        "CFL=0.8, and diffusion step scale 8.\n"
    )
    for label, group in [("Original duration", short), ("Long duration", long)] + [
        (f"Degree {m['parameters']['ell']} / {m['parameters']['orientation']}", m)
        for m in result["mode_controls"]
    ]:
        for issue in group["issues"]:
            lines.append(f"{label}: {issue}\n")
        for row in group["windows"]:
            if row["issues"] and not group["issues"]:
                lo, hi = row["window"]
                lines.append(f"{label}, window {lo:g}--{hi:g}: {'; '.join(row['issues'])}\n")
    for label, audit in result["input_audit"].items():
        lines.append(
            f"Input audit ({label} directory): {len(audit['incomplete'])} incomplete CSV files "
            f"skipped; {len(audit['invalid'])} completed files failed validation. "
            "Paths and reasons are retained in the JSON.\n"
        )
    lines += [text + "\n" for text in result["limitations"]]
    lines += [
        ".. figure:: figures/spherical_control_study.png",
        "   :width: 100%",
        "   :alt: Growth rates and neutral Rayleigh numbers over duration-control fit windows.\n",
        "   Completed matched duration controls. Horizontal bars show the fit-window extent, "
        "not an uncertainty interval. Missing long-run estimates remain unplotted.\n",
    ]
    (output / "spherical_control_study.inc").write_text("\n".join(lines))
    plot(result, output / "figures/spherical_control_study.png", status)


def plot(result, path, status):
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), layout="constrained")
    for source, marker, line in (("long", "o", "-"), ("short", "s", "--")):
        group = result[source]
        centers = np.array([np.mean(w["window"]) for w in group["windows"]])
        widths = np.array([np.diff(w["window"])[0] / 2 for w in group["windows"]])
        for j, ra in enumerate(group["required_Ra"]):
            growth = [
                next((f["growth"] for f in w["fits"] if np.isclose(f["Ra"], ra)), np.nan)
                for w in group["windows"]
            ]
            axes[0].plot(
                centers,
                growth,
                color=f"C{j}",
                marker=marker,
                ls=line,
                markerfacecolor=f"C{j}" if source == "long" else "white",
            )
        roots = [w["neutral"]["Ra"] if w["neutral"] else np.nan for w in group["windows"]]
        axes[1].errorbar(
            centers,
            roots,
            xerr=widths,
            color="C0" if source == "long" else "C3",
            marker=marker,
            ls=line,
            markerfacecolor="C0" if source == "long" else "white",
            label=f"Duration {group['end_diffusive_time']:g}",
            capsize=3,
        )
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[1].axhline(
        result["long"]["spectral_Ra"], color="black", ls=":", label="Spectral reference"
    )
    handles = [
        Line2D([], [], color=f"C{j}", label=f"Ra={ra:.3f} ({factor:g} × reference)")
        for j, (ra, factor) in enumerate(zip(result["long"]["required_Ra"], FACTORS))
    ] + [
        Line2D([], [], color="gray", marker="o", label="Duration 0.6"),
        Line2D(
            [], [], color="gray", marker="s", ls="--", markerfacecolor="white", label="Duration 0.2"
        ),
    ]
    axes[0].legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9, frameon=False
    )
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9, frameon=False)
    axes[0].set(
        title="Modal growth across fit windows",
        ylabel=r"Fitted growth $\widehat{\sigma}=\sigma d^2/\chi_m$",
    )
    axes[1].set(title="Neutral Ra and repeated short windows", ylabel="Bracketed neutral Ra")
    for ax in axes:
        ax.set(xlabel="Fit-window midpoint (diffusive time)", xlim=(0.025, 0.625))
        ax.grid(alpha=0.2)
        if result["long"]["matched_count"] != 3:
            ax.text(
                0.98,
                0.03,
                f"Long triplet incomplete: {result['long']['matched_count']}/3 matched",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=10,
            )
    fig.suptitle(status + "\neta=0.95, degree 43, nr/N=12/132, float64", fontsize=12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HERE / "spherical_samples")
    parser.add_argument("--long-dir", type=Path, help="Default: INPUT_DIR/long_window")
    parser.add_argument("--reference", type=Path, default=HERE / "spherical_reference_results.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-fit-points", type=int, default=10)
    parser.add_argument(
        "--allow-partial", action="store_true", help="Write explicitly partial diagnostics"
    )
    args = parser.parse_args()
    result = analyze(
        args.input_dir, json.loads(args.reference.read_text()), args.long_dir, args.min_fit_points
    )
    result["reference_file"] = str(args.reference.resolve())
    print(
        json.dumps(
            dict(
                status=result["status"],
                long_matches=result["long"]["matched_count"],
                mode_statuses=[m["status"] for m in result["mode_controls"]],
            ),
            indent=2,
        )
    )
    if result["status"] != "control_coverage_complete" and not args.allow_partial:
        raise SystemExit(
            "Supplemental controls incomplete; no artifacts written. "
            "Use --allow-partial for diagnostics."
        )
    write_artifacts(result, args.output_dir)


if __name__ == "__main__":
    main()
