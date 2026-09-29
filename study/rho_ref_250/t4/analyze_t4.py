#!/usr/bin/env python
"""Tables for the #250 T4 matrix written by run_t4.py.

  python analyze_t4.py <rundir> [--json t4.json] > t4_tables.md
"""
import argparse
import glob
import json
import os
import re

import numpy as np

FORMS = ["smooth5", "isentrope", "none", "local_polytrope"]


def read_bin(fname):
    out = []
    with open(fname, "rb") as f:
        while True:
            h = f.read(8)
            if not h:
                break
            nd = int(np.frombuffer(h, np.int64)[0])
            shape = tuple(np.frombuffer(f.read(8 * nd), np.int64))
            n = int(np.prod(shape)) if nd else 1
            out.append(np.frombuffer(f.read(8 * n), np.float64).reshape(shape))
    return out


def summaries(d):
    s = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/summary.*.json"))]
    t = json.load(open(f"{d}/timing.json"))
    return s, t


def assemble(d, kind):
    """Global arrays from per-block files; returns dict of name -> array
    (x3 squeezed, [x2, x1]) plus per-block raw pieces."""
    blocks = [read_bin(f) for f in sorted(glob.glob(f"{d}/{kind}.*.bin"))]
    if not blocks:
        return None, []
    x1 = np.unique(np.concatenate([b[0] for b in blocks]))
    x2 = np.unique(np.concatenate([b[1] for b in blocks]))
    names = (["pref", "dref", "psf", "dsf", "rho", "p"] if kind == "ref"
             else ["rho", "p", "v1", "v2"])
    first = 3 if kind == "ref" else 2
    g = {}
    for k, nm in enumerate(names):
        if nm in ("psf", "dsf"):
            continue  # face arrays are compared per seam, not assembled
        a = np.full((len(x2), len(x1)), np.nan)
        for b in blocks:
            i = np.searchsorted(x1, b[0])
            j = np.searchsorted(x2, b[1])
            a[np.ix_(j, i)] = b[first + k].reshape(len(b[1]), len(b[0]))
        g[nm] = a
    return g, blocks


def seam_jumps(blocks):
    """max relative |dsf| and |psf| mismatch at every shared x1 face."""
    jd = jp = 0.
    nseam = 0
    for a in blocks:
        for b in blocks:
            if a is b or not np.array_equal(a[1], b[1]):
                continue
            if a[2][-1] == b[2][0]:  # a's top face is b's bottom face
                nseam += 1
                psa, dsa = a[5][..., -1], a[6][..., -1]
                psb, dsb = b[5][..., 0], b[6][..., 0]
                scale = np.abs(b[7][..., 0]).max()  # the seam-cell density
                jd = max(jd, np.abs(dsa - dsb).max() / scale)
                jp = max(jp, np.abs(psa - psb).max() / np.abs(psb).max())
    return nseam, jd, jp


def maxrel(a, b):
    return float(np.nanmax(np.abs(a - b)) / max(np.nanmax(np.abs(b)), 1e-300))


def bitequal(g, h):
    return all(np.array_equal(g[k], h[k], equal_nan=True) for k in g)


def seam_table(root):
    rows = []
    for d in sorted(glob.glob(f"{root}/seam_*")):
        if not os.path.exists(f"{d}/timing.json"):
            continue  # still running
        m = re.match(r"seam_(.+?)_(smooth5|isentrope|none|local_polytrope)_"
                     r"(\w+?)_(cpu|gpu)$", os.path.basename(d))
        case, form, tag, dev = m.groups()
        s, t = summaries(d)
        refused = t["rc"] != 0
        row = dict(case=case, form=form, layout=tag, dev=dev, rc=t["rc"],
                   wall=t["wall_s"])
        if "Terminating abnormally" in open(f"{d}/run.log").read():
            row["rc"] = "aborted"
        if not refused:
            ref, rblocks = assemble(d, "ref")
            st, _ = assemble(d, "state")
            row["nseam"], row["dsf_jump"], row["psf_jump"] = \
                seam_jumps(rblocks)
            row["_ref"], row["_state"] = ref, st
        else:
            log = open(f"{d}/run.log").read()
            mm = re.search(r"what\(\):\s+(.*)", log)
            row["why"] = mm.group(1).strip() if mm else "rc != 0"
        rows.append(row)
    # compare with the one-rank run of the same case / form / device
    base = {(r["case"], r["form"], r["dev"]): r for r in rows
            if r["layout"] == "r1" and "_ref" in r}
    cpu = {(r["case"], r["form"]): r for r in rows
           if r["layout"] == "r1" and r["dev"] == "cpu" and "_ref" in r}
    for r in rows:
        b = base.get((r["case"], r["form"], r["dev"]))
        if "_ref" not in r or b is None:
            continue
        r["dref_vs_r1"] = maxrel(r["_ref"]["dref"], b["_ref"]["dref"]) \
            if np.abs(b["_ref"]["dref"]).max() > 0 else \
            float(np.abs(r["_ref"]["dref"]).max())
        r["state_bitequal"] = bitequal(r["_state"], b["_state"])
        r["state_maxrel"] = max(maxrel(r["_state"][k], b["_state"][k])
                                for k in ("rho", "p"))
        r["vel_maxabs"] = max(float(np.abs(r["_state"][k] -
                                           b["_state"][k]).max())
                              for k in ("v1", "v2"))
        c = cpu.get((r["case"], r["form"]))
        if r["dev"] == "gpu" and r["layout"] == "r1" and c is not None:
            r["gpu_vs_cpu_maxrel"] = max(maxrel(r["_state"][k],
                                                c["_state"][k])
                                         for k in ("rho", "p"))
    return rows


def simple_rows(root, prefix):
    rows = []
    for d in sorted(glob.glob(f"{root}/{prefix}_*")):
        if not os.path.exists(f"{d}/timing.json"):
            continue  # still running
        s, t = summaries(d)
        r = dict(name=os.path.basename(d)[len(prefix) + 1:], rc=t["rc"],
                 wall=t["wall_s"])
        if s:
            r.update({k: s[0][k] for k in s[0]})
            r["loop_wall"] = max(x["wall_s"] for x in s)
        # check_redo < 0 prints this and the driver still returns 0
        log = open(f"{d}/run.log").read()
        r["status"] = ("ABORTED (floor redos)" if "Terminating abnormally"
                       in log else ("ok" if t["rc"] == 0 else f"rc={t['rc']}"))
        if r["status"] != "ok":
            r["vmax_over_cs"] = "n/a"
        rows.append(r)
    return rows


def fmt(x):
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.3g}"
    return str(x)


def md(rows, cols, head=None):
    out = ["| " + " | ".join(head or cols) + " |",
           "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    res = {}

    seam = seam_table(args.root)
    print("## Block seams and rank reproducibility\n")
    print(md(seam, ["case", "form", "layout", "dev", "rc", "nseam",
                    "dsf_jump", "psf_jump", "dref_vs_r1", "state_bitequal",
                    "state_maxrel", "vel_maxabs", "gpu_vs_cpu_maxrel",
                    "why"]))
    res["seam"] = [{k: v for k, v in r.items() if not k.startswith("_")}
                   for r in seam]

    for prefix, cols in [
        ("anomaly", ["name", "status", "cycles", "time", "rho_min", "p_min",
                     "nonfinite", "vmax_over_cs", "loop_wall"]),
        ("positivity", ["name", "status", "cycles", "time", "rho_min", "p_min",
                        "nonfinite", "bad_cycle", "vmax_over_cs",
                        "loop_wall"]),
        ("stretch", ["name", "status", "cycles", "vmax_over_cs", "nonfinite",
                     "loop_wall"]),
        ("fault", ["name", "rc", "ref_nonfinite_b0", "fault_nonfinite"]),
        ("cost", ["name", "rc", "cycles", "loop_wall", "ref_us_per_call"]),
    ]:
        rows = simple_rows(args.root, prefix)
        if not rows:
            continue
        print(f"\n## {prefix}\n")
        print(md(rows, cols))
        res[prefix] = rows

    # positivity: guard on/off bit-identical final state
    pairs = []
    for d in sorted(glob.glob(f"{args.root}/positivity_*_guardoff")):
        e = d.replace("_guardoff", "_guardon")
        if not os.path.isdir(e):
            continue
        a, _ = assemble(d, "state")
        b, _ = assemble(e, "state")
        if a is None or b is None:
            continue
        pairs.append(dict(name=os.path.basename(d)[:-len("_guardoff")],
                          bitequal=bitequal(a, b)))
    if pairs:
        print("\n## positivity: guard off vs on\n")
        print(md(pairs, ["name", "bitequal"]))
        res["guard_pairs"] = pairs

    if args.json:
        json.dump(res, open(args.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
