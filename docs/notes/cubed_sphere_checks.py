"""Reproduce the technical note's figures and independent numerical checks.

Run from any directory: python /path/to/cubed_sphere_checks.py
Requires NumPy, SciPy and Matplotlib. This is NOT a Snapy onset simulation.
Analytic ghost values isolate panel discretization from seam exchange errors.
"""
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snapy-note-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np
from scipy.linalg import eigvalsh, eigvals
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"


def metric(alpha, beta):
    x, y = np.tan(alpha), np.tan(beta)
    s = 1 + x*x + y*y
    jac = (1+x*x)*(1+y*y)/s**1.5
    aa = s/(1+x*x)
    bb = s/(1+y*y)
    ab = s*x*y/((1+x*x)*(1+y*y))
    return jac, aa, ab, bb


def xyz(alpha, beta, panel):
    x, y = np.broadcast_arrays(np.tan(alpha), np.tan(beta))
    v = np.stack([np.ones_like(x), x, y]) / np.sqrt(1+x*x+y*y)
    # Orthogonal rotations suffice for the scalar manufacturing experiment.
    return np.roll(v, panel % 3, axis=0) * (1 if panel < 3 else -1)


def solid_angle(edges):
    x, y = np.meshgrid(np.tan(edges), np.tan(edges), indexing="ij")
    primitive = np.arctan2(x*y, np.sqrt(1+x*x+y*y))
    return np.diff(np.diff(primitive, axis=0), axis=1)


def panel_error(n, cross=True):
    h = np.pi/(2*n)
    edges = np.linspace(-np.pi/4, np.pi/4, n+1)
    centers = -np.pi/4 + (np.arange(-1, n+1)+0.5)*h
    a, b = np.meshgrid(centers, centers, indexing="ij")
    area = solid_angle(edges)
    errors, norms = [], []
    for p in range(6):
        v = xyz(a, b, p)
        f = v[0]*v[1]  # degree-two harmonic: Laplace_Omega(f) = -6 f
        ja, aa, ab, _ = metric(edges[:, None], centers[None, 1:-1])
        da = (f[1:, 1:-1]-f[:-1, 1:-1])/h
        db = (f[1:, 2:]-f[1:, :-2]+f[:-1, 2:]-f[:-1, :-2])/(4*h)
        fa = ja*(aa*da + (ab*db if cross else 0))
        jb, _, ab, bb = metric(centers[1:-1, None], edges[None, :])
        db = (f[1:-1, 1:]-f[1:-1, :-1])/h
        da = (f[2:, 1:]-f[:-2, 1:]+f[2:, :-1]-f[:-2, :-1])/(4*h)
        fb = jb*(bb*db + (ab*da if cross else 0))
        numerical = h*(np.diff(fa, axis=0)+np.diff(fb, axis=1))/area
        exact = -6*f[1:-1, 1:-1]
        errors.append(np.sum(area*(numerical-exact)**2))
        norms.append(np.sum(area*exact**2))
    return float(np.sqrt(sum(errors)/sum(norms)))


def critical(epsilon):
    degrees = np.arange(1, int(8/epsilon)+2)
    a2 = epsilon**2*degrees*(degrees+1)
    ra = (np.pi**2+a2)**3/a2
    j = np.argmin(ra)
    return int(degrees[j]), float(ra[j])


def fd_operator(n, a2):
    h = 1/(n+1)
    return (np.diag(np.full(n, 2/h**2+a2))
            + np.diag(np.full(n-1, -1/h**2), 1)
            + np.diag(np.full(n-1, -1/h**2), -1))


def growth(ra, n, a2, pr=1.):
    a = fd_operator(n, a2)
    # w_t = -Pr A w + Pr Ra a^2 A^{-1} theta; theta_t = w - A theta.
    system = np.block([[-pr*a, pr*ra*a2*np.linalg.solve(a, np.eye(n))],
                       [np.eye(n), -a]])
    return float(eigvals(system).real.max())


def main():
    FIG.mkdir(exist_ok=True)
    ns = [12, 24, 48, 96]
    full = [panel_error(n) for n in ns]
    diagonal = [panel_error(n, False) for n in ns]
    orders = np.log2(np.array(full[:-1])/full[1:])
    assert min(orders) > 1.9, orders
    assert diagonal[-1] > 0.1
    assert abs(6*solid_angle(np.linspace(-np.pi/4, np.pi/4, 25)).sum()-4*np.pi) < 1e-12

    eps = 0.05
    ell, exact = critical(eps)
    a2 = eps**2*ell*(ell+1)
    nzs = [15, 31, 63, 127]
    numerical = [float(eigvalsh(fd_operator(n, a2))[0]**3/a2) for n in nzs]
    fd_error = np.abs(np.array(numerical)/exact-1)
    fd_orders = np.log2(fd_error[:-1]/fd_error[1:])
    assert min(fd_orders) > 1.98
    root = brentq(lambda ra: growth(ra, 31, a2), 0.9*exact, 1.1*exact, xtol=1e-8)
    assert abs(root/numerical[1]-1) < 1e-8
    growth_pair = [growth(f*exact, 31, a2) for f in (0.98, 1.02)]
    assert growth_pair[0] < 0 < growth_pair[1]

    # Exact radial resistance flux for T=A+B/r and constant K.
    r = np.linspace(10, 11, 65)
    t = 3 + 2/r
    luminosity_per_sr = -(t[1:]-t[:-1])/(1/r[:-1]-1/r[1:])
    radial_error = float(np.max(np.abs(luminosity_per_sr-2)))
    assert radial_error < 1e-10

    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(121, projection="3d")
    lines = np.linspace(-np.pi/4, np.pi/4, 9)
    curve = np.linspace(-np.pi/4, np.pi/4, 100)
    for p in range(6):
        for c in lines:
            for a, b in [(curve, c), (c, curve)]:
                v = xyz(a, b, p)
                ax.plot(*v, lw=0.5, color=plt.cm.tab10(p))
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title("Six equiangular panels")
    ax = fig.add_subplot(122)
    a, b = np.meshgrid(curve, curve)
    x, y = np.tan(a), np.tan(b)
    cosine = -x*y/np.sqrt((1+x*x)*(1+y*y))
    im = ax.pcolormesh(a, b, cosine, shading="auto", cmap="coolwarm", vmin=-.5, vmax=.5)
    ax.set(xlabel=r"$\alpha$", ylabel=r"$\beta$", title="Cosine of horizontal basis angle", aspect="equal")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "cubed_sphere_geometry.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    a = np.linspace(.8, 4.5, 400)
    axs[0].plot(a, (np.pi**2+a*a)**3/(a*a), label="continuous wavenumber")
    ls = np.arange(20, 76)
    al = eps*np.sqrt(ls*(ls+1))
    axs[0].plot(al, (np.pi**2+al*al)**3/(al*al), ".", label=r"$d/R=0.05$, integer $\ell$")
    axs[0].set(xlabel=r"$a=k_h d$", ylabel="Neutral Rayleigh number", ylim=(640, 1400))
    axs[0].legend(fontsize=8)
    ras = np.linspace(.96*exact, 1.04*exact, 41)
    axs[1].plot(ras, [growth(v, 31, a2) for v in ras], label="FD, 31 interior points")
    axs[1].axhline(0, color="gray", lw=.7)
    axs[1].axvline(exact, color="black", ls="--", label="analytical thin shell")
    axs[1].set(xlabel="Rayleigh number", ylabel=r"Growth rate ($\chi/d^2$)")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "convection_onset.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].loglog(ns, full, "o-", label="full metric")
    axs[0].loglog(ns, diagonal, "s-", label="cross terms omitted")
    axs[0].set(xlabel="Cells per panel edge", ylabel="Relative weighted L2 error")
    axs[0].legend()
    axs[1].loglog(np.array(nzs)+1, fd_error, "o-")
    axs[1].set(xlabel="Radial intervals", ylabel="Relative critical Ra error")
    for ax, ticks in zip(axs, (ns, np.array(nzs)+1)):
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
    fig.tight_layout()
    fig.savefig(FIG / "diffusion_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    results = dict(scope="Independent Boussinesq/metric checks; NOT a Snapy onset run",
                   panel_n=ns, panel_error=full, panel_order=orders.tolist(),
                   diagonal_metric_error=diagonal, epsilon=eps, ell=ell,
                   analytical_Ra=exact, radial_interior_points=nzs,
                   numerical_Ra=numerical, onset_order=fd_orders.tolist(),
                   growth_root=root, growth_at_098_102=growth_pair,
                   radial_resistance_error=radial_error,
                   shell_minima=[dict(epsilon=e, ell=critical(e)[0], Ra=critical(e)[1])
                                 for e in (.2, .1, .05, .02, .01)])
    (HERE / "cubed_sphere_results.json").write_text(json.dumps(results, indent=2)+"\n")
    rows = [".. Generated by cubed_sphere_checks.py; do not edit.\n",
            ".. list-table:: Independent radial eigenproblem (d/R = 0.05, ell = %d)" % ell,
            "   :header-rows: 1\n", "   * - Interior points", "     - Numerical critical Ra",
            "     - Relative error"]
    for n, ra, err in zip(nzs, numerical, fd_error):
        rows.extend([f"   * - {n}", f"     - {ra:.8f}", f"     - {err:.4e}"])
    rows.extend(["", ".. list-table:: Panel metric manufactured solution (analytic ghosts)",
                 "   :header-rows: 1\n", "   * - Cells per edge", "     - Full metric error",
                 "     - Diagonal-only error"])
    for n, good, bad in zip(ns, full, diagonal):
        rows.extend([f"   * - {n}", f"     - {good:.5e}", f"     - {bad:.5e}"])
    (HERE / "cubed_sphere_tables.inc").write_text("\n".join(rows)+"\n")
    print(json.dumps(results, indent=2))
    print("PASS: metric convergence, sphere area, onset eigenproblem, growth bracket, radial conduction")


if __name__ == "__main__":
    main()
