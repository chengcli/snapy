"""Does the ONE-SIDED x1 pressure difference preserve rest balance and
telescoping, including at the first and last interior cell?

d1_pressure (hydro_forward.cpp, Zoey's 0b6b6ef): centred where both x1
neighbours exist, one-sided at is=il and ie=iu so no x1 ghost row is read.
The flux's p* and the geometric source's p* must use the SAME difference.
At rest p is x2-independent, so the face-reconstructed p equals the cell p
exactly and the two call sites see identical input.
"""
import numpy as np
np.seterr(all="raise")

GRAV, RD, T0, P0, NG = 11.0, 3700.0, 100.0, 1.0e5, 3


def d1_pressure(p, x1v, is_, ie):
    """exact transcription of the coded stencil, interior indices 1..n1-2"""
    n1 = len(p)
    d = (p[2:] - p[:-2]) / (x1v[2:] - x1v[:-2])          # centred, idx 1..n1-2
    d[is_ - 1] = (p[is_ + 1] - p[is_]) / (x1v[is_ + 1] - x1v[is_])   # one-sided
    d[ie - 1] = (p[ie] - p[ie - 1]) / (x1v[ie] - x1v[ie - 1])        # one-sided
    return d


def delta(rm, rp):
    h, rb = rp - rm, .5 * (rm + rp)
    h2, t = h * h, 12. * rb * rb
    return h2 * (t - h2) / (12. * rb * (t + h2))


def run(nz=24, nth=12, r0=7.0e7, depth=4.0e4, th0=1.20, th1=1.50, dphi=0.4):
    nc1 = nz + 2 * NG
    rf = np.linspace(r0 - NG * depth / nz, r0 + depth + NG * depth / nz, nc1 + 1)
    rm, rp = rf[:-1], rf[1:]
    x1v = .75 * (rp ** 4 - rm ** 4) / (rp ** 3 - rm ** 3)
    p = P0 * np.exp(-(x1v - x1v[NG]) * GRAV / (RD * T0))
    is_, ie = NG, NG + nz - 1

    thf = np.linspace(th0, th1, nth + 1)
    tm, tp = thf[:-1], thf[1:]
    radial = .5 * (rp ** 2 - rm ** 2)
    radial_volume = (rp ** 3 - rm ** 3) / 3.
    polar_volume = np.abs(np.cos(tm) - np.cos(tp))
    A2 = radial[None, :] * np.abs(np.sin(thf))[:, None] * dphi
    V = radial_volume[None, :] * polar_volume[:, None] * dphi
    S = (radial / radial_volume)[None, :] * \
        ((np.abs(np.sin(tp)) - np.abs(np.sin(tm))) / polar_volume)[:, None]

    d1p = d1_pressure(p, x1v, is_, ie)
    ds = delta(rm, rp)[1:-1]
    pstar = p.copy()
    pstar[1:-1] = p[1:-1] - ds * d1p

    def resid(pf, psrc):
        lat = (A2[1:, :] * pf[None, :] - A2[:-1, :] * pf[None, :]) / V
        return lat - S * psrc[None, :]

    sl = slice(is_, ie + 1)
    scale = np.abs(S[:, sl] * p[None, sl]).max()
    out = {}
    for name, pf, psrc in (("base", p, p),
                           ("flux only", pstar, p),
                           ("flux+source (coded)", pstar, pstar)):
        r = np.abs(resid(pf, psrc)[:, sl])
        out[name] = (r.max(), r.max() / scale, r[:, 0].max() / scale,
                     r[:, -1].max() / scale)
    print("  x2-momentum rest residual, interior x1 cells only")
    print("  %-22s %-13s %-11s %-11s %-11s" %
          ("arm", "max abs", "rel max", "rel @first", "rel @last"))
    for k, v in out.items():
        print("  %-22s %-13.6e %-11.3e %-11.3e %-11.3e" % (k, *v))
    print(f"  source scale |S p| = {scale:.6e}")
    print(f"  one-sided ends: d1p[first]={d1p[is_-1]:+.6e}  "
          f"centred neighbour={((p[is_+2]-p[is_])/(x1v[is_+2]-x1v[is_])):+.6e}")
    print(f"                  d1p[last] ={d1p[ie-1]:+.6e}  "
          f"centred neighbour={((p[ie+1]-p[ie-1])/(x1v[ie+1]-x1v[ie-1])):+.6e}")
    # telescoping of the correction across the x2 column (closed walls)
    corr = -ds * d1p
    tel = ((A2[1:, sl] - A2[:-1, sl]) * corr[None, sl]).sum(axis=0)
    print(f"  x2 telescoping of the correction, max |sum over j| = "
          f"{np.abs(tel).max():.6e}  relative "
          f"{np.abs(tel).max()/np.abs((A2[1:,sl]*corr[None,sl])).max():.3e}")


print("=== SPHERICAL-POLAR, one-sided x1 pressure difference ===")
run()
