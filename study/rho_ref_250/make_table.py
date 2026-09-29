#!/usr/bin/env python
"""Markdown tables from analyze_t3.py --json output.

  python make_table.py t3.json > tables.md
"""
import json
import sys

FORMS = ["smooth5", "isentrope", "none"]
BUBBLE = {
    "straka": ["theta_p_min", "u_max", "u_min", "w_max", "w_min", "front_m",
               "theta_p_max"],
    "bryan": ["theta_e_p_max", "theta_e_p_min", "w_max", "w_min",
              "theta_rho_p_max", "theta_rho_p_min", "top_m(theta_e'=1K)"],
}
REST = {
    "straka": ["max|w|", "max|u|", "max|dtheta|"],
    "bryan": ["max|w|", "max|u|", "max|dtheta_e|", "max|dtheta_rho|"],
}


def fmt(x):
    if x is None:
        return "-"
    a = abs(x)
    if a != 0 and (a < 1e-2 or a >= 1e5):
        return f"{x:.3g}"
    return f"{x:.2f}" if a >= 100 else f"{x:.3f}"


def main():
    data = json.load(open(sys.argv[1]))
    ref, rows = data["ref"], data["rows"]
    gpu = [r for r in rows if r["device"] != "cpu"]
    out = []
    worst = {}
    for case in ["straka", "bryan"]:
        keys = BUBBLE[case]
        out.append(f"\n#### {case}: bubble, metric (rel. error vs published)\n")
        out.append("| dx | form | " + " | ".join(k.replace("|", "\\|") for k in keys) + " | worst rel err | wall s |")
        out.append("|---" * (len(keys) + 4) + "|")
        out.append("| ref | - | " + " | ".join(
            fmt(ref[case].get(k)) if k in ref[case] else "n/a" for k in keys)
            + " | - | - |")
        runs = sorted([r for r in gpu if r["case"] == case and not r["rest"]],
                      key=lambda r: (-r["dx"], FORMS.index(r["form"])))
        for r in runs:
            m = r["metrics"]
            if m is None:
                out.append(f"| {r['dx']} | {r['form']} | FAILED rc={r['rc']} |")
                continue
            cells, rels = [], []
            for k in keys:
                v = m.get(k)
                if k in ref[case]:
                    rel = abs(v - ref[case][k]) / abs(ref[case][k])
                    rels.append((rel, k))
                    cells.append(f"{fmt(v)} ({100 * rel:.1f}%)")
                else:
                    cells.append(fmt(v))
            w = max(rels)
            worst.setdefault((case, r["dx"]), {})[r["form"]] = w
            out.append(f"| {r['dx']} | {r['form']} | " + " | ".join(cells) +
                       f" | {100 * w[0]:.1f}% ({w[1]}) | {r['wall_s']:.0f} |")
        keys = REST[case]
        out.append(f"\n#### {case}: unperturbed background (initial-state "
                   "residual at t_end)\n")
        out.append("| dx | form | " + " | ".join(k.replace("|", "\\|") for k in keys) + " | wall s |")
        out.append("|---" * (len(keys) + 3) + "|")
        runs = sorted([r for r in gpu if r["case"] == case and r["rest"]],
                      key=lambda r: (-r["dx"], FORMS.index(r["form"])))
        for r in runs:
            m = r["metrics"] or {}
            out.append(f"| {r['dx']} | {r['form']} | " + " | ".join(
                fmt(m.get(k)) for k in keys) + f" | {r['wall_s']:.0f} |")

    out.append("\n#### worst-case relative error per form (max over the "
               "referenced metrics)\n")
    out.append("| case | dx | " + " | ".join(FORMS) + " | best | ratio worst/best |")
    out.append("|---" * 6 + "|")
    for (case, dx), d in sorted(worst.items(), key=lambda t: (t[0][0], -t[0][1])):
        vals = {f: d[f][0] for f in FORMS if f in d}
        best = min(vals, key=vals.get)
        out.append(f"| {case} | {dx} | " + " | ".join(
            f"{100 * d[f][0]:.1f}% ({d[f][1]})" if f in d else "-"
            for f in FORMS) + f" | {best} | " + ", ".join(
            f"{f} {vals[f] / vals[best]:.2f}" for f in vals) + " |")

    cpu = [r for r in rows if r["device"] == "cpu"]
    if cpu:
        out.append("\n#### CPU spot checks (max |metric_cpu - metric_gpu|)\n")
        out.append("| run | max abs diff over metrics | cpu wall s | gpu wall s |")
        out.append("|---|---|---|---|")
        for c in cpu:
            g = next((r for r in gpu if r["name"] == c["name"][:-3] + "gpu"),
                     None)
            if not g or not c["metrics"] or not g["metrics"]:
                continue
            d = max(abs(c["metrics"][k] - g["metrics"][k])
                    for k in c["metrics"])
            out.append(f"| {c['name'][:-4]} | {d:.3g} | {c['wall_s']:.0f} | "
                       f"{g['wall_s']:.0f} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
