#!/usr/bin/env python3
"""Gate: the moist EOS "UT->I" path must agree with "W->I".

`compute("UT->I", [u, T])` and `compute("W->I", [w])` return the same physical
quantity -- internal energy per unit volume, reference offset included. The
temperature-floor clamp in `EquationOfStateImpl::apply_conserved_limiter_`
depends on that: it forms `min_ie` by calling "UT->I" at the configured
`temperature-floor` and clamps `cons[IPR]` to `ke + min_ie`. If "UT->I"
disagrees, the floor silently enforces some other temperature.

Two independent checks, so a regression that breaks one shape still trips:

  (1) ROUND TRIP. For a state w at its own temperature T, "UT->I" must
      reproduce "W->I" over a temperature sweep.

  (2) AFFINE STRUCTURE. ie(T) must be affine in T -- the reference offset u0 is
      an energy per unit mass and belongs to the intercept, while the heat
      capacity (dry channel included) is the slope. A term in the wrong place
      shows up as a bent line or as an intercept that is not the offset.

Both were violated before the fix: the u0 offsets were multiplied by T, and the
dry heat-capacity term rho_dry*cv_dry*T was absent entirely, which with a
realistic ice-giant species set made "UT->I" return 1.1% of the true internal
energy and turned a 20 K floor into a 0.23 K one.

  python test_eos_temp2inteng.py [--device cuda] [--yaml PATH]
"""
import argparse
import os
import sys

import torch


def build(yaml_file: str, device: str):
    from snapy import MeshBlock, MeshBlockOptions

    op = MeshBlockOptions.from_yaml(yaml_file)
    block = MeshBlock(op)
    block.to(torch.device(device))

    eos = None
    for name, m in block.named_modules():
        if type(m).__name__ == "IdealMoist":
            eos = m
            break
    assert eos is not None, "config must select an ideal-moist EOS"
    return block, eos


def make_state(block, eos, device):
    """A mostly-dry carrier with a few percent condensible, uniform over the block."""
    from snapy import kICY

    bufs = dict(block.named_buffers())
    w = bufs["hydro.D"].clone().zero_()  # primitive scratch, (nvar, nc3, nc2, nc1)
    ny = w.size(0) - kICY
    assert ny >= 1, f"config must register at least one species, got ny={ny}"

    w[0] = 1.0  # IDN: total density
    w[4] = 1.0e5  # IPR
    w[kICY + 0] = 0.05  # a vapour
    if ny > 1:
        w[kICY + 1] = 0.01  # and a condensate
    return w


def set_temperature(eos, w, target):
    """W->T is linear in p at fixed density and composition, so one probe calibrates it."""
    w[4] = 1.0e5
    t1 = eos.compute("W->T", [w]).mean().item()
    w[4] = 1.0e5 * (target / t1)
    return eos.compute("W->T", [w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="test_eos_temp2inteng.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rtol", type=float, default=1.0e-12)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    yaml_file = args.yaml
    if not os.path.exists(yaml_file):
        yaml_file = os.path.join(here, args.yaml)

    block, eos = build(yaml_file, args.device)
    w = make_state(block, eos, args.device)

    failures = []

    # ---- (1) round trip against W->I ---------------------------------------
    print(f"{'T [K]':>10} {'W->I [J/m3]':>18} {'UT->I [J/m3]':>18} {'rel err':>12}")
    for target in (50.0, 100.0, 200.0, 400.0):
        T = set_temperature(eos, w, target)
        u = eos.compute("W->U", [w])
        ie_w = eos.compute("W->I", [w])
        ie_ut = eos.compute("UT->I", [u, T])

        err = (ie_ut - ie_w).abs().max().item()
        scale = ie_w.abs().max().item()
        rel = err / scale if scale > 0 else err
        print(f"{T.mean().item():10.3f} {ie_w.mean().item():18.10e} "
              f"{ie_ut.mean().item():18.10e} {rel:12.3e}")
        if not (rel <= args.rtol):
            failures.append(
                f"round trip at T={target} K: UT->I and W->I differ by "
                f"{rel:.3e} (rtol {args.rtol:.1e})"
            )

    # ---- (2) ie(T) must be affine, with the offset as the intercept --------
    # Three temperatures at equal spacing: an affine ie has equal increments.
    T0, dT = 100.0, 100.0
    ies = []
    for k in range(3):
        T = set_temperature(eos, w, T0 + k * dT)
        u = eos.compute("W->U", [w])
        # hold the composition fixed in CONSERVED space so the slope is a pure
        # dT response: same u, only T varies
        if k == 0:
            u_ref = u.clone()
        ies.append(eos.compute("UT->I", [u_ref, T]).mean().item())

    d1, d2 = ies[1] - ies[0], ies[2] - ies[1]
    curv = abs(d2 - d1) / max(abs(d1), abs(d2), 1.0)
    print(f"\nie(T) increments over dT={dT} K: {d1:.10e}, {d2:.10e} "
          f"-> curvature {curv:.3e}")
    if not (curv <= 1.0e-12):
        failures.append(
            f"ie(T) is not affine in T: increments {d1:.6e} vs {d2:.6e} "
            f"(curvature {curv:.3e}); a term that is not a heat capacity is "
            f"being scaled by temperature"
        )

    # intercept: extrapolate to T=0 and compare against the reference offset
    slope = d1 / dT
    intercept = ies[0] - slope * T0
    ebufs = dict(eos.named_buffers())
    u0 = ebufs["u0"].double()
    from snapy import kICY

    ny = u_ref.size(0) - kICY
    offset = (u_ref[0] * u0[0]).mean().item() + sum(
        (u_ref[kICY + i] * u0[1 + i]).mean().item() for i in range(ny)
    )
    off_scale = max(abs(offset), abs(intercept), 1.0)
    off_err = abs(intercept - offset) / off_scale
    print(f"ie(T=0) extrapolated = {intercept:.10e}, "
          f"reference offset = {offset:.10e}, rel err {off_err:.3e}")
    if not (off_err <= 1.0e-10):
        failures.append(
            f"the T=0 intercept of ie(T) is {intercept:.6e} but the reference "
            f"offset is {offset:.6e} (rel err {off_err:.3e}); u0 is an energy "
            f"per unit mass and must not be scaled by temperature"
        )

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        sys.exit(1)
    print("PASS: UT->I agrees with W->I, and ie(T) is affine with the offset "
          "as its intercept")


if __name__ == "__main__":
    main()
