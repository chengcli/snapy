#!/usr/bin/env python3
"""Derived caches after a conversion of a child module, and memos under autograd.

Each check compares a warm module (caches populated, then the operation) with a fresh one that took the
same operation before any call, so both end with the same buffers and any difference is a stale cache:

  cubed:   a 6-face cubed-sphere mesh. The coordinate and hydro children of every block are round-tripped
           (float64 -> float32 -> float64); the hydro child's own hook drops the Hydro and implicit caches,
           so what is left is the layout's velocity transforms, built from the coordinate's angles.
           The u after one RK stage is compared.
  mixture: the moist-mixture EOS. Its ivol/temp memo is keyed on the caller's primitive state, which a
           conversion of the EOS does not touch, while ivol and temp are buffers that it does convert.
           W->A and W->T are compared after an EOS round trip.
  grad:    the ideal-moist temperature memo with a primitive state that needs gradients: a second
           backward pass on the same state must work and give the fresh gradient. The moist-mixture EOS
           does not support gradients (its memo writes into buffers with set_); that must stay an
           explicit error, not a silent value.

A float32 base round-trips through float64, which is exact: those round-trip arms are controls.

  python test_cache_child_conversion.py [--device cpu|cuda]
"""
import argparse
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_flux_covariance_rows as rows  # noqa: E402  (cubed-sphere card and builders)

F64, F32 = torch.float64, torch.float32

MIXTURE = """
reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: dry, composition: {O: 0.42, N: 1.56, Ar: 0.01}, cv_R: 2.5}
  - {name: H2O, composition: {H: 2, O: 1}, cv_R: 2.5, u0_R: 0.}
  - {name: H2O(l), composition: {H: 2, O: 1}, cv_R: 9.0, u0_R: -3430.}
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 7, nx2: 5, nx3: 1, nghost: 3}
dynamics:
  equation-of-state: {type: %s, density-floor: 1.e-10, pressure-floor: 1.e-10, limiter: false}
boundary-condition:
  external: {x1-inner: reflecting, x1-outer: reflecting, x2-inner: periodic, x2-outer: periodic,
             x3-inner: periodic, x3-outer: periodic}
reactions:
  - {equation: H2O <=> H2O(l), type: nucleation, rate-constant: {formula: h2o_ideal}}
"""


def other(dtype):
    return F32 if dtype == F64 else F64


def same(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)


def scale(a, b):
    if isinstance(a, list):
        return max(scale(x, y) for x, y in zip(a, b))
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()


# ---------------------------------------------------------------------------
def cubed_mesh(device, dtype):
    cfg = rows.card("gnomonic-equiangle", moist=False)
    mesh, blocks = rows.blocks_of(cfg, device)  # float64: balanced_w needs it
    nx1 = cfg["geometry"]["cells"]["nx1"]

    def wind(b, w):
        x, y, z = (b.buffer("coord.x%dv" % i).cpu().double() for i in (2, 3, 1))
        xx = ((x - x[0]) / (x[-1] - x[0]))[None, :, None]
        yy = ((y - y[0]) / (y[-1] - y[0]))[:, None, None]
        zz = ((z - z[0]) / (z[-1] - z[0]))[None, None, :]
        w[2] = (5. * torch.sin(math.pi * xx) * (2 * zz - 1)).expand_as(w[2]).to(w)
        w[3] = (5. * torch.cos(2 * math.pi * yy) * zz).expand_as(w[3]).to(w)

    w0 = [rows.balanced_w(b, nx1, wind=wind).to(device, dtype) for b in blocks]
    for b in blocks:
        b.to(device, dtype)
    return mesh, blocks, w0


def cubed_stage(mesh, w0):
    mv, _ = mesh.initialize([{"hydro_w": w.clone()} for w in w0])
    mesh.set_cycle(1)
    mesh.forward(mv, 5.0, 0)
    return [v["hydro_u"].detach().cpu().clone() for v in mv]


def cubed_case(device, dtype, children):
    def round_trip(blocks):
        for b in blocks:
            for c in children:
                b.module(c).to(other(dtype))
                b.module(c).to(dtype)

    mesh, blocks, w0 = cubed_mesh(device, dtype)
    cubed_stage(mesh, w0)
    round_trip(blocks)
    got = cubed_stage(mesh, w0)
    mesh2, blocks2, w02 = cubed_mesh(device, dtype)
    round_trip(blocks2)
    want = cubed_stage(mesh2, w02)
    return got, want


# ---------------------------------------------------------------------------
def eos_block(device, dtype, eos):
    fd, path = tempfile.mkstemp(suffix=".yaml", dir=os.getcwd())
    os.write(fd, (MIXTURE % eos).encode())
    os.close(fd)
    try:
        import snapy
        b = snapy.MeshBlock(snapy.MeshBlockOptions.from_yaml(path))
    finally:
        os.unlink(path)
    b.to(device, dtype)
    return b


def prim(b, device, dtype):
    w = torch.zeros_like(b.buffer("hydro.D")).to(device, dtype)
    n1 = w.size(-1)
    ramp = torch.linspace(0, 1, n1, dtype=dtype, device=device)
    w[0] = 1.1
    w[4] = 1.e5 * (1 + 0.1 * ramp)
    w[5] = 0.01 + 0.003 * ramp
    w[6] = 0.002
    return w


def mixture_case(device, dtype, ab):
    b = eos_block(device, dtype, "moist-mixture")
    eos = b.module("hydro.eos")
    w = prim(b, device, dtype)
    eos.compute(ab, [w])
    eos.to(other(dtype))
    eos.to(dtype)
    got = eos.compute(ab, [w]).clone()
    f = eos_block(device, dtype, "moist-mixture")
    feos = f.module("hydro.eos")
    feos.to(other(dtype))
    feos.to(dtype)
    want = feos.compute(ab, [prim(f, device, dtype)]).clone()
    return got, want


def dependency_case(device, dtype, eos_name, operation):
    b = eos_block(device, dtype, eos_name)
    eos = b.module("hydro.eos")
    w = prim(b, device, dtype)
    eos.compute("W->T", [w])
    if operation == "thermo_round_trip":
        thermo = b.module("hydro.eos.thermo")
        thermo.to(other(dtype))
        thermo.to(dtype)
    elif operation == "weight_edit":
        eos.options.weight(eos.options.weight() * 1.01)
    elif operation == "thermo_edit":
        b.buffer("hydro.eos.thermo.inv_mu").mul_(1.01)
    else:
        b.buffer("hydro.eos.inv_mu_ratio_m1").add_(0.01)
    got = eos.compute("W->T", [w]).clone()
    # An equal but distinct primitive forces recomputation with the same final
    # dependency buffers. No tolerance or baseline numerical value is changed.
    want = eos.compute("W->T", [w.clone()]).clone()
    return got, want


def species_grad_case(device, dtype):
    b = eos_block(device, dtype, "ideal-moist")
    eos = b.module("hydro.eos")
    w = prim(b, device, dtype)
    eos.compute("W->T", [w])  # warm before enabling the dependency's gradient
    table = b.buffer("hydro.eos.inv_mu_ratio_m1").requires_grad_(True)
    eos.compute("W->T", [w]).sum().backward()
    table.grad = None
    eos.compute("W->T", [w]).sum().backward()
    got = table.grad.clone()
    f = eos_block(device, dtype, "ideal-moist")
    ft = f.buffer("hydro.eos.inv_mu_ratio_m1").requires_grad_(True)
    f.module("hydro.eos").compute("W->T", [prim(f, device, dtype)]).sum().backward()
    return got, ft.grad.clone()


def grad_case(device, dtype):
    b = eos_block(device, dtype, "ideal-moist")
    eos = b.module("hydro.eos")
    w = prim(b, device, dtype).requires_grad_(True)
    eos.compute("W->T", [w]).sum().backward()
    w.grad = None
    eos.compute("W->T", [w]).sum().backward()  # the same state again
    got = w.grad.clone()
    f = eos_block(device, dtype, "ideal-moist")
    wf = prim(f, device, dtype).requires_grad_(True)
    f.module("hydro.eos").compute("W->T", [wf]).sum().backward()
    return got, wf.grad.clone()


def mixture_grad_is_explicit(device, dtype):
    b = eos_block(device, dtype, "moist-mixture")
    w = prim(b, device, dtype).requires_grad_(True)
    try:
        b.module("hydro.eos").compute("W->T", [w]).sum().backward()
    except RuntimeError as e:
        return "derivative for set_ is not implemented" in str(e), str(e).splitlines()[0]
    return False, "no error"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device: skipped")
        return 125
    device = torch.device("cuda:0" if a.device == "cuda" else "cpu")
    failures = []

    def check(name, fn):
        try:
            got, want = fn()
        except RuntimeError as e:
            failures.append(name)
            print("FAIL %-48s raised: %s" % (name, str(e).splitlines()[0][:100]), flush=True)
            return
        if same(got, want):
            print("ok   %-48s bitwise" % name, flush=True)
        else:
            failures.append(name)
            print("FAIL %-48s max|warm-fresh|/max|fresh| = %.3e" % (name, scale(got, want)), flush=True)

    for dtype, tag in ((F64, "f64"), (F32, "f32 (control)")):
        check("cubed coord+hydro %s %s" % (a.device, tag), lambda: cubed_case(device, dtype, ("coord", "hydro")))
        check("cubed coord %s %s" % (a.device, tag), lambda: cubed_case(device, dtype, ("coord",)))
        for ab in ("W->A", "W->T"):
            check("mixture %s %s %s" % (ab, a.device, tag), lambda: mixture_case(device, dtype, ab))
        for eos_name, operation in (("moist-mixture", "thermo_round_trip"),
                                    ("moist-mixture", "thermo_edit"),
                                    ("ideal-moist", "species_edit"),
                                    ("ideal-moist", "weight_edit")):
            check("%s %s %s %s" % (eos_name, operation, a.device, tag),
                  lambda: dependency_case(device, dtype, eos_name, operation))
        check("species gradient %s %s" % (a.device, tag), lambda: species_grad_case(device, dtype))
        name = dtype == F64 and "f64" or "f32"
        check("grad ideal-moist W->T %s %s" % (a.device, name), lambda: grad_case(device, dtype))
        raised, why = mixture_grad_is_explicit(device, dtype)
        if raised:
            print("ok   %-48s explicit error: %s" % ("grad moist-mixture %s %s" % (a.device, name), why[:80]))
        else:
            failures.append("grad moist-mixture")
            print("FAIL grad moist-mixture %s %s: no explicit error" % (a.device, name))

    if failures:
        print("FAILED:", len(failures), failures)
        return 1
    print("### cache child conversion test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
