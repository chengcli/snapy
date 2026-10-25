"""Independent Chebyshev stability calculation for compressible planar onset.

Ideal gas; constant dynamic viscosity and Fourier conductivity; impermeable,
stress-free, fixed-temperature walls. Equations are the dimensional equivalent
of Alboussiere & Ricard (2017), JFM 817, 264--305, sections 5 and 8.1.
Our units are d = T_mid = rho_mid = R = 1. No snapy discretization is used.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.linalg import eig, eigvals
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent


def chebyshev(n):
    """n+1 Lobatto nodes on [-1/2, 1/2], in decreasing order."""
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1)
    c[[0, -1]] = 2
    c *= (-1.0) ** np.arange(n + 1)
    delta = x[:, None] - x[None, :]
    D = (c[:, None] / c[None, :]) / (delta + np.eye(n + 1))
    D -= np.diag(D.sum(axis=1))
    return x / 2, 2 * D


def background(z, a, dissipation, gamma):
    cp = gamma / (gamma - 1)
    g = cp * dissipation
    temperature = 1 - a * z
    rho = temperature ** (g / a - 1)
    drho = -(g - a) * rho / temperature
    return temperature, rho, drho, g, cp


def matrices(n, k, a, dissipation, gamma, ra=None):
    """A x = lambda B x. x=(U,w,theta,q), physical u_x=i U.

    With ra=None, lambda=Ra_sa, theta=g*T'/mu and q=p'/mu.
    Otherwise lambda=sigma in physical time, theta=T' and q=p'.
    The time-dependent problem retains the density time derivative.
    """
    z, D = chebyshev(n)
    m = len(z)
    I, Z = np.eye(m), np.zeros((m, m))
    T, rho, drho, g, cp = background(z, a, dissipation, gamma)
    r, rt, it = np.diag(rho), np.diag(rho / T), np.diag(1 / T)
    lap = D @ D - k * k * I
    h = D @ D - (4 / 3) * k * k * I
    v = (4 / 3) * D @ D - k * k * I
    if ra is None:
        A = np.block(
            [
                [-k * r, r @ D + np.diag(drho), Z, Z],
                [h, k / 3 * D, Z, -k * I],
                [-k / 3 * D, v, rt, -D - g * it],
                [Z, Z, lap, Z],
            ]
        )
        B = np.zeros_like(A)
        B[3 * m : 4 * m, m : 2 * m] = -r
    else:
        chi = np.sqrt(g * (a - dissipation) / ra)
        mu, conductivity = chi, cp * chi  # Pr=1
        A = np.block(
            [
                [k * r, -r @ D - np.diag(drho), Z, Z],
                [mu * h, mu * k / 3 * D, Z, -k * I],
                [-mu * k / 3 * D, mu * v, g * rt, -D - g * it],
                [Z, cp * (a - dissipation) * r, conductivity * lap, Z],
            ]
        )
        B = np.block([[Z, Z, -rt, it], [r, Z, Z, Z], [Z, r, Z, Z], [Z, Z, cp * r, -I]])
    # Replace momentum/thermal endpoint equations, leaving continuity intact.
    for j in (0, m - 1):
        for row, variable, boundary in (
            (m + j, 0, D[j]),
            (2 * m + j, 1, I[j]),
            (3 * m + j, 2, I[j]),
        ):
            A[row] = 0
            B[row] = 0
            A[row, variable * m : (variable + 1) * m] = boundary
    return A, B


def neutral(n, k, a, dissipation, gamma=5 / 3):
    A, B = matrices(n, k, a, dissipation, gamma)
    ev = eigvals(A, B)
    good = np.isfinite(ev) & (ev.real > 0) & (np.abs(ev.imag) < 1.0e-6)
    return float(ev[good].real.min())


def critical(n, a, dissipation, gamma=5 / 3):
    result = minimize_scalar(
        lambda k: neutral(n, k, a, dissipation, gamma),
        bounds=(1.5, 3.0),
        method="bounded",
        options={"xatol": 1.0e-7},
    )
    assert result.success
    return dict(Ra=float(result.fun), k=float(result.x))


def growth(n, k, a, dissipation, ra, gamma=5 / 3):
    A, B = matrices(n, k, a, dissipation, gamma, ra)
    values, vectors = eig(A, B)
    good = np.isfinite(values) & (np.abs(values) < 1.0e8)
    values, vectors = values[good], vectors[:, good]
    index = np.argmax(values.real)
    sigma = values[index]
    vec = vectors[:, index]
    residual = np.linalg.norm(A @ vec - sigma * (B @ vec)) / (
        np.linalg.norm(A) * np.linalg.norm(vec)
        + abs(sigma) * np.linalg.norm(B) * np.linalg.norm(vec)
    )
    assert residual < 1.0e-9
    _, _, _, g, _ = background(np.array([0.0]), a, dissipation, gamma)
    chi = np.sqrt(g * (a - dissipation) / ra)
    return dict(
        real=float(sigma.real / chi), imag=float(sigma.imag / chi), residual=float(residual)
    )


def two_mode(a, dissipation, gamma=5 / 3):
    """Published second-order approximation, equation (8.9), not exact.

    The corresponding equation is (8.11) in the arXiv manuscript.
    """
    t = dissipation / (1 - 1 / gamma)
    p = np.pi
    return float(
        27 * p**4 / 4
        + (2 * p * p - 320 / 39) * t * t
        + (9 * p**4 / 8 - 17 * p * p / 4 + 512 / 13) * a * t
        - (27 * p**4 / 16 - 63 * p * p / 8 + 1216 / 39) * a * a
    )


def main():
    z, D = chebyshev(24)
    assert np.max(np.abs(D @ (z**3) - 3 * z * z)) < 1.0e-10
    boussinesq = 27 * np.pi**4 / 4
    weak = critical(32, 0.001, 0.0002)
    assert abs(weak["Ra"] - boussinesq) < 2.0e-4
    rows = []
    for name, a, dissipation in (("weak", 0.1, 0.02), ("stratified", 1.0, 0.5)):
        levels = [dict(N=n, **critical(n, a, dissipation)) for n in (24, 32, 48)]
        assert abs(levels[-1]["Ra"] - levels[-2]["Ra"]) < 2.0e-5
        best = levels[-1]
        rates = [
            dict(
                Ra=factor * best["Ra"], **growth(48, best["k"], a, dissipation, factor * best["Ra"])
            )
            for factor in (0.98, 1.0, 1.02)
        ]
        assert rates[0]["real"] < 0 < rates[-1]["real"]
        assert abs(rates[1]["real"]) < 2.0e-6
        rows.append(
            dict(
                name=name,
                a=a,
                D=dissipation,
                gamma=5 / 3,
                levels=levels,
                two_mode=two_mode(a, dissipation),
                growth=rates,
            )
        )
    result = dict(
        reference="Alboussiere & Ricard (2017), DOI 10.1017/jfm.2017.108",
        convention="d=T_mid=rho_mid=R=1; Pr=1; Ra_sa=g*(a-D)/(nu_mid*chi_mid)",
        boussinesq=boussinesq,
        weak_limit=weak,
        cases=rows,
    )
    (HERE / "planar_reference_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
