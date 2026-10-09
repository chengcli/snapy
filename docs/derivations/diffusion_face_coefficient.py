"""Face coefficient of the x1 diffusive flux with an x1 profile: product of
means against mean of products.

A transcription of DiffusionImpl::forward's x1 conductive flux on a uniform
Cartesian column with reflecting walls: interior face coefficient from the two
neighbouring cells, wall face extrapolated linearly from the two nearest active
cells, normal gradient (T_i - T_{i-1}) / dx, tendency -(F_{i+1/2} - F_{i-1/2})
/ dx. The coefficient is c = s * q with s the x1 profile and q the cell value
(rho, or rho cv); both forms extrapolate the product s q at a wall.

  1. constant c (s = 1/q, T linear): the mean of products gives zero tendency
     to round-off; the product of means gives an O(dx^2) tendency.
  2. smooth variable c: both forms give a second-order face flux.

  python diffusion_face_coefficient.py
"""
import numpy as np

NG = 2


def rho_of(x):
    """polytrope-like column: rho falls ~10x over [0, 1], fastest at the top"""
    return (1.25 - x) ** 1.5


def column(nx):
    dx = 1. / nx
    x = (np.arange(nx + 2 * NG) - NG + 0.5) * dx
    return x, dx


def face_coefficient(s, q, form):
    """interior faces i - 1/2 for i = NG .. NG + nx (nx + 1 faces)"""
    a, b = slice(NG - 1, -NG), slice(NG, -NG + 1 or None)
    if form == "product_of_means":
        c = 0.5 * (s[a] + s[b]) * 0.5 * (q[a] + q[b])
    else:
        c = 0.5 * (s[a] * q[a] + s[b] * q[b])
    p = s * q  # walls: linear extrapolation of the product, uniform mesh
    c[0] = 1.5 * p[NG] - 0.5 * p[NG + 1]
    c[-1] = 1.5 * p[-NG - 1] - 0.5 * p[-NG - 2]
    return c


def face_flux(s, q, T, dx, form):
    a, b = slice(NG - 1, -NG), slice(NG, -NG + 1 or None)
    return -face_coefficient(s, q, form) * (T[b] - T[a]) / dx


def tendency(s, q, T, dx, form):
    F = face_flux(s, q, T, dx, form)
    return -(F[1:] - F[:-1]) / dx


def main():
    print("1. constant c = s q = 1, T = 300 + 50 x, F = -50: max|tendency| over all cells,")
    print("   and over the interior (the two wall cells left out)")
    print("%6s %14s %14s %8s %14s %8s" % ("nx", "mean_of_prod", "prod_of_means", "order",
                                         "interior", "order"))
    prev = None
    for nx in (16, 32, 64, 128):
        x, dx = column(nx)
        q = rho_of(x)
        s, T = 1. / q, 300. + 50. * x
        new = np.abs(tendency(s, q, T, dx, "mean_of_products")).max()
        du = tendency(s, q, T, dx, "product_of_means")
        old, inner = np.abs(du).max(), np.abs(du[1:-1]).max()
        order = ("", "") if prev is None else tuple(
            "%.2f" % np.log2(a / b) for a, b in zip(prev, (old, inner)))
        print("%6d %14.3e %14.3e %8s %14.3e %8s" % (nx, new, old, order[0], inner, order[1]))
        prev = old, inner
        assert new < 1e-12 * 50. / dx, new

    print("2. smooth c = (1 + 0.5 cos 3x) rho, T = 300 + 50 sin 2x: "
          "max|F - F_exact| / max|F_exact| at the faces")
    print("%6s %14s %8s %14s %8s" % ("nx", "mean_of_prod", "order", "prod_of_means", "order"))
    prev = {}
    for nx in (32, 64, 128, 256):
        x, dx = column(nx)
        xf = np.arange(nx + 1) * dx
        q, s, T = rho_of(x), 1. + 0.5 * np.cos(3. * x), 300. + 50. * np.sin(2. * x)
        exact = -(1. + 0.5 * np.cos(3. * xf)) * rho_of(xf) * 100. * np.cos(2. * xf)
        row = [nx]
        for form in ("mean_of_products", "product_of_means"):
            e = np.abs(face_flux(s, q, T, dx, form) - exact).max() / np.abs(exact).max()
            row += [e, "" if form not in prev else "%.2f" % np.log2(prev[form] / e)]
            prev[form] = e
        print("%6d %14.3e %8s %14.3e %8s" % tuple(row))
        if row[2]:
            assert float(row[2]) > 1.9, row


if __name__ == "__main__":
    main()
