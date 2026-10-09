"""Hydrostatic rest balance of the x2-momentum row, with and without the
centroid shift, on the spherical-polar and gnomonic cubed-sphere grids.

Every discrete expression is transcribed from the source at head
10270ed57aca82ecc758904938e54fa88d672947:
  spherical_polar.cpp:185-191 face_area2, :200-211 cell_volume,
                      :89-104 coord_src1_i / coord_src1_j, :262-266 the source
  gnomonic_equiangle.cpp:220-229 cell_volume, :212-215 face_area2,
                      :83-87 cosine/sine_face2_kj, :136-141 x_ov_rD_kji,
                      :445-452 src2
  coordinate.cpp:492-496 the x2 divergence, :504 /vol
At rest (u == 0) the only nonzero x2-face momentum flux is the pressure, so the
row is a cancellation of (lateral pressure flux divergence) against (geometric
pressure source), BOTH LINEAR in the same cell pressure p.
"""
import numpy as np

np.seterr(all="raise")

# ---------------------------------------------------------------- hydrostatics
GRAV, RD, T0, P0 = 11.0, 3700.0, 100.0, 1.0e5


def column(rm, rp):
    """Isothermal hydrostatic p on the cell centres of a radial grid."""
    rv = 0.75 * (rp ** 4 - rm ** 4) / (rp ** 3 - rm ** 3)      # volume centroid
    return rv, P0 * np.exp(-(rv - rv[0]) * GRAV / (RD * T0))


def delta(rm, rp):
    """r_v - r_c, exact and cancellation-free (coordinate.cpp radial_face_centroid_shift_)."""
    h, rbar = rp - rm, 0.5 * (rm + rp)
    h2, t = h * h, 12.0 * rbar * rbar
    return h2 * (t - h2) / (12.0 * rbar * (t + h2))


def shift(rv, p, rm, rp):
    """p* = p - delta * dp/dr, centred difference on rv, interior only."""
    d = delta(rm, rp)
    dp = np.zeros_like(p)
    dp[1:-1] = (p[2:] - p[:-2]) / (rv[2:] - rv[:-2])
    return p - d * dp, dp


# ------------------------------------------------------------------- spherical
def spherical(nr=24, nth=12, r0=7.0e7, depth=4.0e4, th0=1.20, th1=1.50, dphi=0.4):
    rf = np.linspace(r0, r0 + depth, nr + 1)
    rm, rp = rf[:-1], rf[1:]
    thf = np.linspace(th0, th1, nth + 1)
    tm, tp = thf[:-1], thf[1:]
    rv, p = column(rm, rp)

    radial = 0.5 * (rp ** 2 - rm ** 2)                    # :186-187
    radial_volume = (rp ** 3 - rm ** 3) / 3.0             # :201-203
    polar_volume = np.abs(np.cos(tm) - np.cos(tp))        # :206-208
    sin_m, sin_p = np.abs(np.sin(tm)), np.abs(np.sin(tp))
    # face_area2 = radial * |sin theta_f| * dphi  (:188-190)
    A2 = radial[None, :] * np.abs(np.sin(thf))[:, None] * dphi
    V = radial_volume[None, :] * polar_volume[:, None] * dphi
    src1_i = radial / radial_volume                       # :89-91
    src1_j = (sin_p - sin_m) / polar_volume               # :95-97

    def residual(pflux, psrc):
        # coordinate.cpp:492-496 then :504 ; spherical_polar.cpp:266
        lateral = (A2[1:, :] * pflux[None, :] - A2[:-1, :] * pflux[None, :]) / V
        source = src1_i[None, :] * src1_j[:, None] * psrc[None, :]
        return lateral - source

    ps, dp = shift(rv, p, rm, rp)
    return {
        "base": residual(p, p),
        "flux only": residual(ps, p),
        "flux+source": residual(ps, ps),
        "scale": float(np.max(np.abs(src1_i[None, :] * src1_j[:, None] * p[None, :]))),
        "coef_identity": float(np.max(np.abs(
            (A2[1:, :] - A2[:-1, :]) / V - src1_i[None, :] * src1_j[:, None]))),
        "coef_scale": float(np.max(np.abs(src1_i[None, :] * src1_j[:, None]))),
        "dp": dp, "delta": delta(rm, rp), "rv": rv, "p": p,
    }


# -------------------------------------------------------------------- gnomonic
def gnomonic(nr=24, na=8, r0=7.0e7, depth=4.0e4):
    rf = np.linspace(r0, r0 + depth, nr + 1)
    rm, rp = rf[:-1], rf[1:]
    rv_mid = 0.5 * (rm + rp)                        # gnomonic x1v is the midpoint
    af = np.linspace(-0.25 * np.pi, 0.25 * np.pi, na + 1)
    av = 0.5 * (af[:-1] + af[1:])
    rv, p = column(rm, rp)                          # exact volume centroid for p

    x, xf = np.tan(av), np.tan(af)
    y = np.tan(av)
    C, Cf, D = np.sqrt(1 + x * x), np.sqrt(1 + xf * xf), np.sqrt(1 + y * y)
    # :83-84  (k index = x3 -> y, j index = x2 -> x)
    cos_f2 = -xf[None, :] * y[:, None] / (Cf[None, :] * D[:, None])
    sin_f2 = np.sqrt(1 + xf[None, :] ** 2 + y[:, None] ** 2) / (Cf[None, :] * D[:, None])
    # dx3f_ang_face2_kj :119 : angle between the two x3 faces at the x2 face
    y1, y2 = np.tan(af[:-1]), np.tan(af[1:])
    d1 = np.sqrt(1 + xf[None, :] ** 2 + y1[:, None] ** 2)
    d2 = np.sqrt(1 + xf[None, :] ** 2 + y2[:, None] ** 2)
    ang_f2 = np.arccos((1 + xf[None, :] ** 2 + y1[:, None] * y2[:, None]) / (d1 * d2))
    # face_area2 = (x1v * dx1f) * dx3f_ang_face2_kj  (:212-214)
    A2 = (rv_mid * (rp - rm))[None, None, :] * ang_f2[:, :, None]
    # exact solid angle (:124-136) and cell_volume (:220-229)
    corner = np.arctan(xf[None, :] * xf[:, None]
                       / np.sqrt(1 + xf[None, :] ** 2 + xf[:, None] ** 2))
    sa = corner[1:, 1:] - corner[1:, :-1] - corner[:-1, 1:] + corner[:-1, :-1]
    radial = (rp - rm) * (rp * rp + rp * rm + rm * rm) / 3.0
    V = radial[None, None, :] * sa[:, :, None]
    fx = A2 * sin_f2[:, :, None]                               # :136
    x_ov_rD = (fx[:, 1:, :] - fx[:, :-1, :]) / V                # :137-138

    def residual(pflux, psrc):
        # at rest flux2[IVY] = p * sine_face2_kj after flux2global2_
        lateral = (fx[:, 1:, :] * pflux[None, None, :]
                   - fx[:, :-1, :] * pflux[None, None, :]) / V
        source = x_ov_rD * psrc[None, None, :]
        return lateral - source

    ps, dp = shift(rv, p, rm, rp)
    return {
        "base": residual(p, p),
        "flux only": residual(ps, p),
        "flux+source": residual(ps, ps),
        "scale": float(np.max(np.abs(x_ov_rD * p[None, None, :]))),
        "coef_identity": 0.0,   # identical by construction, see note
        "coef_scale": float(np.max(np.abs(x_ov_rD))),
        "dp": dp, "delta": delta(rm, rp), "rv": rv, "p": p,
    }


for name, fn in (("SPHERICAL-POLAR", spherical), ("GNOMONIC CUBED-SPHERE", gnomonic)):
    R = fn()
    print(f"=== {name} ===")
    print(f"  source scale |S p|                 {R['scale']:.6e}")
    print(f"  geometric coefficient identity     max|(dA2)/V - S_geom| = {R['coef_identity']:.3e}"
          f"   (coef scale {R['coef_scale']:.6e})")
    for arm in ("base", "flux only", "flux+source"):
        a = np.abs(R[arm])
        rel = a.max() / R["scale"]
        print(f"  {arm:12s} max|resid| {a.max():.6e}   relative {rel:.3e}")
    d, dp = R["delta"], R["dp"]
    i = len(dp) // 2
    print(f"  delta[mid] {d[i]:.6e} m   dp/dr[mid] {dp[i]:.6e}   "
          f"delta*dp/dr {d[i]*dp[i]:.6e} Pa   p {R['p'][i]:.6e} Pa")
    print(f"  relative pressure shift delta*|dp/dr|/p = {abs(d[i]*dp[i])/R['p'][i]:.3e}")
    print()
