#!/usr/bin/env python
"""#250 decision rule over T1, T2 step 3 and T3.

For every regime row (a regime at one resolution), divide each form's error
by the best form's error on that row; a form's score is its largest ratio.

  python decide_250.py --t1 t1_comment.md --t2 t2_comment.md --t3 t3.json \
      [--max-dzh 0.5] [--json out.json]

--t1 / --t2 are the T1 and T2-step-3 comment bodies on the issue (their
tables are parsed as posted, not re-run); --t3 is analyze_t3.py's JSON.
"""
import argparse
import json
import re

FORMS = ["smooth5", "isentrope", "none", "local_polytrope"]


def t1_rows(path, max_dzh):
    rows = {}
    pat = re.compile(r"^(\S+) \| (\w+) \| ([0-9.e+-]+) \| [0-9.e+-]+ \| "
                     r"([0-9.e+-]+)\s*$")
    for line in open(path):
        m = pat.match(line.strip())
        if m and float(m.group(3)) <= max_dzh:
            key = ("T1 " + m.group(1), float(m.group(3)))
            rows.setdefault(key, {})[m.group(2)] = float(m.group(4))
    return rows


def t2_rows(path):
    rows = {}
    pat = re.compile(r"^\| (\w+) \| ([0-9.]+) \| \d+ \| ([-+0-9.e]+) \|")
    for line in open(path):
        m = pat.match(line.strip())
        if m:
            key = ("T2 isothermal gravity mode |rel_freq|", float(m.group(2)))
            rows.setdefault(key, {})[m.group(1)] = abs(float(m.group(3)))
    return rows


def t3_rows(path):
    data = json.load(open(path))
    ref, rows = data["ref"], {}
    for r in data["rows"]:
        if r["device"] == "cpu" or r["rest"]:
            continue
        key = ("T3 " + r["case"], float(r["dx"]))
        if r["metrics"] is None:
            rows.setdefault(key, {})[r["form"]] = float("inf")
            continue
        err = max(abs(r["metrics"][k] - v) / abs(v)
                  for k, v in ref[r["case"]].items())
        rows.setdefault(key, {})[r["form"]] = err
    return rows


def worst(rows, forms):
    out = {}
    for f in forms:
        best = (0., None)
        for key, e in rows.items():
            if not all(g in e for g in forms):
                continue
            b = min(e[g] for g in forms)
            r = e[f] / b if b > 0 else (1. if e[f] == 0 else float("inf"))
            if r > best[0]:
                best = (r, key)
        out[f] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1")
    ap.add_argument("--t2")
    ap.add_argument("--t3")
    ap.add_argument("--max-dzh", type=float, default=0.5)
    ap.add_argument("--json", default="")
    ap.add_argument("--by-regime", action="store_true",
                    help="also print each regime's worst ratio per form")
    args = ap.parse_args()
    sets = {}
    if args.t1:
        sets["T1"] = t1_rows(args.t1, args.max_dzh)
    if args.t2:
        sets["T2"] = t2_rows(args.t2)
    if args.t3:
        sets["T3"] = t3_rows(args.t3)
    allrows = {k: v for s in sets.values() for k, v in s.items()}
    sets["all"] = allrows
    res = {}
    print("| test set | rows | " + " | ".join(FORMS) + " | smallest worst |")
    print("|---" * (len(FORMS) + 3) + "|")
    for name, rows in sets.items():
        w = worst(rows, FORMS)
        pick = min(FORMS, key=lambda f: w[f][0])
        res[name] = {f: {"ratio": w[f][0], "where": w[f][1]} for f in FORMS}
        res[name]["pick"] = pick
        cells = [f"{w[f][0]:.3g} ({w[f][1][0]}, {w[f][1][1]:g})"
                 for f in FORMS]
        print(f"| {name} | {len(rows)} | " + " | ".join(cells) +
              f" | {pick} |")
    if args.by_regime:
        print("\n| regime | resolutions | " + " | ".join(FORMS) + " |")
        print("|---" * (len(FORMS) + 2) + "|")
        for reg in sorted({k[0] for k in allrows}):
            rows = {k: v for k, v in allrows.items() if k[0] == reg}
            w = worst(rows, FORMS)
            res.setdefault("by_regime", {})[reg] = {f: w[f][0] for f in FORMS}
            res_s = ", ".join(f"{k[1]:g}" for k in sorted(rows))
            print(f"| {reg} | {res_s} | " +
                  " | ".join(f"{w[f][0]:.3g} (at {w[f][1][1]:g})"
                             if w[f][1] else "-" for f in FORMS) + " |")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
