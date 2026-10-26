"""Independent fully compressible, nonrotating spherical-shell onset reference.

Units: d = T_mid = rho_mid = gas constant R = 1. The radius ratio is eta=ri/ro,
not the fractional thickness. Gravity is constant inward, g=cp*dissipation.
Dynamic viscosity mu and Fourier conductivity K are spatially constant, with
zero bulk viscosity (Stokes hypothesis). Walls are impermeable, stress free,
and fixed temperature. No snapy operator, Boussinesq, or anelastic approximation
is used. The stationary constraint follows by setting sigma=0 in continuity.

BACKGROUND
----------
ri=eta/(1-eta), ro=1/(1-eta), rm=(ri+ro)/2, z=r-rm. Set B=a*ri*ro and
A=1-B/rm, so T=A+B/r=1-B*z/(r*rm), T(ri)-T(ro)=a, T(rm)=1.
p=exp(-g*J), rho=p/T, where J=integral_rm^r ds/T(s). For A != 0,
J=z/A-B/A**2*log1p(A*z/rm). At A=0, J=z+z**2/(2*rm).
The implementation evaluates an algebraically equivalent cancellation-safe
expression, including a series near A*z/rm=0. In particular rho(rm)=p(rm)=1.
The thermal driving H=-dT/dr-dissipation=B/r**2-dissipation varies radially.

VECTOR SPHERICAL HARMONICS AND THE FOUR FIELDS
-------------------------------------------
Let L=ell*(ell+1), s=sqrt(L), Delta_Omega Y=-L*Y, and use
u=w(r)*Y*e_r + U(r)*grad_Omega(Y)/s, with exp(sigma*t) dependence.
Scalar perturbations are theta(r)*Y and q(r)*Y. Thus the coefficient of the
UNNORMALIZED unit-sphere gradient is V=U/s. No extra factor of r is implicit.
The eigenvalues do not depend on azimuthal order m. Define

  delta = w' + 2*w/r - s*U/r,
  S f = f'' + 2*f'/r - L*f/r**2,
  F_h = S U + 2*s*w/r**2 + s*delta/(3*r),
  F_r = S w - 2*w/r**2 + 2*s*U/r**2 + delta'/3.

Here F is div(tau)/mu = vector_laplacian(u) + grad(div(u))/3. The full
physical growth pencil, with theta=Tpert and q=ppert, is

  sigma*(q/T-rho*theta/T) = -rho*delta - rho'*w,
  sigma*rho*U = -s*q/r + mu*F_h,
  sigma*rho*w = -q' - g*q/T + g*rho*theta/T + mu*F_r,
  sigma*(rho*cp*theta-q) = rho*cp*H*w + K*S theta.

At stationary onset rescale theta=g*Tpert/mu and q=ppert/mu. The pencil is

  0 = rho*delta + rho'*w,
  0 = -s*q/r + F_h,
  0 = -q' - g*q/T + rho*theta/T + F_r,
  S theta = -Ra*rho*H/(a-dissipation)*w.

BC at both walls: w=0, U'-U/r=0, theta=0. The full tangential traction is
mu*(U'-U/r+s*w/r); impermeability reduces it to the BC written above.
Ra=g*(a-dissipation)/(nu_mid*chi_mid), chi_mid=K/cp, nu_mid=mu, Pr=nu/chi.
Growth is returned in thermal diffusion units sigma/chi_mid as well as
physical units sigma. Toroidal modes decouple and cannot drive convection;
the ell=1 rigid-rotation neutral toroidal mode is therefore excluded. ell=0
is a separate radial acoustic/thermal sector, not a convective overturning
mode. The critical search covers poloidal ell>=1.

DISCRETIZATION / PRESSURE ARTIFACT
--------------------------------
Use degree-n U,w,theta and degree-(n-2) pressure, with continuity at the
interior Lobatto nodes. The equal-degree nodal pencil has a spurious pressure
branch: its damping tends to zero as n grows, despite a tiny matrix residual.
The mixed-degree pencil removes this branch. Additionally, growth candidates
must satisfy the independently evaluated differential equations on a dense
off-grid mesh; rejected unresolved modes are recorded, not hidden by a rate
cutoff. A regression reproduces the old artifact at eta=.5, ell=3, Ra=820
and checks the corrected pressure and growth across n=24,32,48,64.

PRIMARY SOURCES / SCOPE
----------------------
The thermodynamic linearization and planar comparison follow Alboussiere &
Ricard (2017), JFM 817, 264-305, DOI 10.1017/jfm.2017.108, sections 5 and 8.1:
https://arxiv.org/pdf/1607.02193
The angular convention is the radial/spheroidal decomposition documented by
SHTns, with its spheroidal coefficient S_lm=U/s:
https://nschaeff.bitbucket.io/shtns/vsh.html
The spherical radial equations and numerical results here are independently
derived/computed; they are not a published spherical critical-Rayleigh table.

Run the complete CPU reference and its assertions:
  OPENBLAS_NUM_THREADS=1 python docs/notes/spherical_compressible_reference.py
Export a growing/decaying mode at the specified physical Rayleigh number:
  OPENBLAS_NUM_THREADS=1 python docs/notes/spherical_compressible_reference.py \
    --export-profile /tmp/shell-profile.txt --eta .9 --ell 21 --ra 700 --nr 32
The headerless file contains exactly nr ascending cell-center rows
z,w,U,Tpert,ppert, normalized together to max(abs(w))=chi_mid; the runner
multiplies ALL FOUR perturbations by its small seed amplitude.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import eig, eigvals
from scipy.optimize import minimize_scalar
from scipy.fft import dct
from numpy.polynomial.chebyshev import chebder, chebval, chebvander

HERE = Path(__file__).resolve().parent
ETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.9999)


def chebyshev(n):
    """Degree-n Lobatto collocation on z in [-1/2,1/2], descending nodes."""
    if int(n) != n or n < 8:
        raise ValueError("Chebyshev degree must be an integer >=8")
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1)
    c[[0, -1]] = 2
    c *= (-1.0) ** np.arange(n + 1)
    delta = x[:, None] - x[None, :]
    derivative = (c[:, None] / c[None, :]) / (delta + np.eye(n + 1))
    derivative -= np.diag(derivative.sum(axis=1))
    return x / 2, 2 * derivative


def geometry(eta):
    if not 0 < eta < 1:
        raise ValueError("eta=ri/ro must lie strictly between zero and one")
    ri, ro = eta / (1 - eta), 1 / (1 - eta)
    return ri, ro, (ri + ro) / 2


def background(z, eta, a=1.0, dissipation=0.5, gamma=5 / 3):
    """Return exact backgrounds/derivatives at z=r-rm (arrays or scalar).

    Dictionary keys include r,T,rho,p,dT,drho,H,g,cp,ri,ro,rm,A,B,J.
    z rather than absolute r preserves radial accuracy for very thin shells.
    """
    z = np.asarray(z, dtype=float)
    ri, ro, rm = geometry(eta)
    if gamma <= 1 or not a > dissipation > 0:
        raise ValueError("Require gamma>1 and a>dissipation>0")
    cp = gamma / (gamma - 1)
    g = cp * dissipation
    B = a * ri * ro
    A = 1 - B / rm
    r = rm + z
    T = 1 - (B / rm) * z / r
    if np.any(r <= 0) or np.any(T <= 0):
        raise ValueError("Background radius and temperature must be positive")
    x = A * z / rm
    small = np.abs(x) < 1.0e-3
    h = np.empty_like(x)
    # (x-log(1+x))/x**2 = sum_{j>=0} (-x)**j/(j+2).
    h[small] = np.polynomial.polynomial.polyval(x[small], [(-1.0) ** j / (j + 2) for j in range(8)])
    h[~small] = (x[~small] - np.log1p(x[~small])) / x[~small] ** 2
    J = z + (1 - A) * z**2 / rm * h
    p = np.exp(-g * J)
    rho = p / T
    dT = -B / r**2
    drho = rho * (-g - dT) / T
    return dict(
        r=r,
        T=T,
        rho=rho,
        p=p,
        dT=dT,
        drho=drho,
        H=-dT - dissipation,
        g=g,
        cp=cp,
        ri=ri,
        ro=ro,
        rm=rm,
        A=A,
        B=B,
        J=J,
    )


def transport(ra, a=1.0, dissipation=0.5, gamma=5 / 3, pr=1.0):
    """Constant mu,K and midpoint nu,chi for the given superadiabatic Ra."""
    if ra <= 0 or pr <= 0 or not a > dissipation > 0 or gamma <= 1:
        raise ValueError("Require Ra,Pr>0, gamma>1, and a>dissipation>0")
    cp = gamma / (gamma - 1)
    chi = np.sqrt(cp * dissipation * (a - dissipation) / (pr * ra))
    return dict(chi=float(chi), nu=float(pr * chi), mu=float(pr * chi), K=float(cp * chi))


def matrices(
    n, eta, ell, a=1.0, dissipation=0.5, gamma=5 / 3, ra=None, pr=1.0, reduced_pressure=True
):
    """A x=lambda B x, x=(U,w,theta,q); neutral lambda=Ra or growth sigma.

    A real ell>=1 is allowed ONLY to locate the continuous minimum. Actual
    spherical modes use integer ell. theta,q scaling is stated in module docs.
    U,w,theta have degree n and nodal unknowns. By default q has degree n-2
    and Chebyshev COEFFICIENT unknowns; continuity is enforced at the n-1
    interior nodes. This mixed-degree discretization removes the unresolved
    pressure branch of the equal-degree nodal pencil. Set reduced_pressure
    False only to reproduce that artifact or test the nodal operator blocks.
    """
    if ell < 1:
        raise ValueError("Poloidal degree ell must be >=1")
    z, derivative = chebyshev(n)
    bg = background(z, eta, a, dissipation, gamma)
    m = len(z)
    identity, zero = np.eye(m), np.zeros((m, m))
    inverse_r = 1 / bg["r"]
    k = np.sqrt(ell * (ell + 1)) * inverse_r
    kr = np.diag(k)
    rho = np.diag(bg["rho"])
    rho_over_T = np.diag(bg["rho"] / bg["T"])
    inverse_T = np.diag(1 / bg["T"])
    scalar_lap = derivative @ derivative + np.diag(2 * inverse_r) @ derivative - kr @ kr
    div_h, div_r = -kr, derivative + np.diag(2 * inverse_r)
    coupling = np.diag(2 * k * inverse_r)
    visc_hh = scalar_lap + kr @ div_h / 3
    visc_hr = coupling + kr @ div_r / 3
    visc_rh = coupling + derivative @ div_h / 3
    visc_rr = scalar_lap - np.diag(2 * inverse_r**2) + derivative @ div_r / 3
    mass_h = rho @ div_h
    mass_r = rho @ div_r + np.diag(bg["drho"])
    pressure_r = -derivative - bg["g"] * inverse_T
    if ra is None:
        A = np.block(
            [
                [mass_h, mass_r, zero, zero],
                [visc_hh, visc_hr, zero, -kr],
                [visc_rh, visc_rr, rho_over_T, pressure_r],
                [zero, zero, scalar_lap, zero],
            ]
        )
        B = np.zeros_like(A)
        B[3 * m : 4 * m, m : 2 * m] = -np.diag(bg["rho"] * bg["H"] / (a - dissipation))
    else:
        tr = transport(ra, a, dissipation, gamma, pr)
        mu = tr["mu"]
        A = np.block(
            [
                [-mass_h, -mass_r, zero, zero],
                [mu * visc_hh, mu * visc_hr, zero, -kr],
                [mu * visc_rh, mu * visc_rr, bg["g"] * rho_over_T, pressure_r],
                [zero, np.diag(bg["rho"] * bg["cp"] * bg["H"]), tr["K"] * scalar_lap, zero],
            ]
        )
        B = np.block(
            [
                [zero, zero, -rho_over_T, inverse_T],
                [rho, zero, zero, zero],
                [zero, rho, zero, zero],
                [zero, zero, bg["cp"] * rho, -identity],
            ]
        )
    # Continuity is retained at all nodes. Replace momentum/thermal wall rows.
    for j in (0, m - 1):
        for row, variable, condition in (
            (m + j, 0, derivative[j] - inverse_r[j] * identity[j]),
            (2 * m + j, 1, identity[j]),
            (3 * m + j, 2, identity[j]),
        ):
            A[row] = 0
            B[row] = 0
            A[row, variable * m : (variable + 1) * m] = condition
    if reduced_pressure:
        pressure_basis = chebvander(2 * z, n - 2)
        rows = np.ones(4 * m, dtype=bool)
        rows[[0, m - 1]] = False
        A = np.column_stack((A[:, : 3 * m], A[:, 3 * m :] @ pressure_basis))[rows]
        B = np.column_stack((B[:, : 3 * m], B[:, 3 * m :] @ pressure_basis))[rows]
    return A, B


def _residual(A, B, value, vector):
    return float(
        np.linalg.norm(A @ vector - value * (B @ vector))
        / ((np.linalg.norm(A) + abs(value) * np.linalg.norm(B)) * np.linalg.norm(vector))
    )


def neutral(n, eta, ell, a=1.0, dissipation=0.5, gamma=5 / 3):
    """Lowest positive real stationary Rayleigh eigenvalue at fixed ell."""
    A, B = matrices(n, eta, ell, a, dissipation, gamma)
    # At ell=1 in a very thin shell k is tiny and Ra scales as k**-2.
    # Balance U~1/k, theta,q~1/k**2 and solve for lambda=Ra*k**2.
    # Without this exact change of variables QZ can classify every low-k
    # eigenvalue as infinite. Row equilibration also balances the tau BCs.
    m = n + 1
    k = np.sqrt(ell * (ell + 1)) / geometry(eta)[2]
    columns = np.repeat([1 / k, 1, 1 / k**2, 1 / k**2], [m, m, m, m - 2])
    rows = np.repeat([1, k, k**2, k**2], [m - 2, m, m, m])
    A = rows[:, None] * A * columns[None, :]
    B = rows[:, None] * B * columns[None, :] / k**2
    row_norm = np.max(np.abs(A), axis=1)
    A /= row_norm[:, None]
    B /= row_norm[:, None]
    values = eigvals(A, B, check_finite=False)
    good = np.isfinite(values) & (values.real > 0) & (np.abs(values.imag) < 1.0e-7)
    if not np.any(good):
        raise RuntimeError("No positive real stationary eigenvalue")
    return float(np.min(values[good].real) / k**2)


def critical(n, eta, a=1.0, dissipation=0.5, gamma=5 / 3):
    """Find integer-ell minimum, with a broad sampled continuous-k audit.

    k=sqrt(ell*(ell+1))/rm. Scan from ell=1 to k=16 (or a larger endpoint),
    locate every sampled local minimum, refine each in k, and check neighboring
    integers. The audit records sampled branches, endpoints, and neighbors;
    this is a numerical global-bracket check, not a proof about ell=infinity.
    Small integer ranges are additionally exhaustively checked (<=128 modes).
    """
    _, _, rm = geometry(eta)
    kmin = np.sqrt(2) / rm
    kmax = max(16.0, 4 * kmin)

    def degree(k):
        return max(1.0, (np.sqrt(1 + 4 * (k * rm) ** 2) - 1) / 2)

    cache = {}

    def at_k(k):
        key = float(k)
        if key not in cache:
            cache[key] = neutral(n, eta, degree(k), a, dissipation, gamma)
        return cache[key]

    grid = np.unique(
        np.concatenate(
            (
                np.geomspace(kmin, kmax, 33),
                np.linspace(max(kmin, 0.75), min(kmax, 5.0), 23),
            )
        )
    )
    vals = np.array([at_k(k) for k in grid])
    minima = [j for j in range(1, len(grid) - 1) if vals[j] <= min(vals[j - 1], vals[j + 1])]
    candidates = {1}
    continuous = []
    for j in minima:
        result = minimize_scalar(
            at_k,
            bounds=(grid[j - 1], grid[j + 1]),
            method="bounded",
            options={"xatol": 2.0e-7},
        )
        if not result.success:
            raise RuntimeError("Continuous minimization failed")
        ell = degree(result.x)
        continuous.append(dict(ell=float(ell), k=float(result.x), Ra=float(result.fun)))
        candidates.update(range(max(1, int(ell) - 2), int(ell) + 4))
    if not continuous:
        if vals[0] != np.min(vals):
            raise RuntimeError("Global scan failed to bracket a minimum")
        continuous.append(dict(ell=1.0, k=float(kmin), Ra=float(vals[0])))
    ellmax = int(degree(kmax))
    if ellmax <= 128:
        candidates.update(range(1, ellmax + 1))
    integer = [
        dict(ell=ell, Ra=neutral(n, eta, ell, a, dissipation, gamma)) for ell in sorted(candidates)
    ]
    best = min(integer, key=lambda item: item["Ra"])
    if not vals[-1] > 10 * best["Ra"] or not vals[-1] > vals[-2] > vals[-3]:
        raise RuntimeError("High-degree tail not safely outside the minimum")
    return dict(
        N=int(n),
        eta=float(eta),
        ell=int(best["ell"]),
        Ra=float(best["Ra"]),
        k=float(np.sqrt(best["ell"] * (best["ell"] + 1)) / rm),
        continuous=min(continuous, key=lambda item: item["Ra"]),
        integer_candidates=integer,
        global_bracket=dict(
            k_min=float(kmin),
            k_max=float(kmax),
            ell_max=ellmax,
            all_integers_checked=ellmax <= 128,
            sampled_k=grid.tolist(),
            sampled_Ra=vals.tolist(),
            local_minima=len(minima),
            high_k_ratio=float(vals[-1] / best["Ra"]),
        ),
    )


def _offgrid_residual(n, eta, ell, ra, sigma, vector, a, dissipation, gamma, pr):
    """Evaluate the DIFFERENTIAL equations between collocation points.

    This deliberately does not reuse the assembled matrices. Differentiate
    Chebyshev interpolants analytically and evaluate every physical term on
    257 (or 4*n+1) Gauss nodes, including near the walls. Each equation's error
    is max|sum terms| / sum(max|term|). Matrix eigenpair residuals alone miss
    the equal-degree pressure checkerboard artifact.
    """
    count = max(257, 4 * n + 1)
    x = np.cos(np.pi * (np.arange(count) + 0.5) / count)
    bg = background(x / 2, eta, a, dissipation, gamma)
    coeff = dct(vector.reshape(4, n + 1), type=1, axis=1) / n
    coeff[:, [0, -1]] *= 0.5
    U, w, theta, q = chebval(x, coeff.T)
    dU, dw, dtheta, dq = 2 * chebval(x, chebder(coeff.T, axis=0))
    ddU, ddw, ddtheta, _ = 4 * chebval(x, chebder(coeff.T, 2, axis=0))
    r, rho, T = bg["r"], bg["rho"], bg["T"]
    L = ell * (ell + 1)
    s = np.sqrt(L)
    div = dw + 2 * w / r - s * U / r
    ddiv = ddw + 2 * dw / r - 2 * w / r**2 - s * dU / r + s * U / r**2
    force_h = ddU + 2 * dU / r - L * U / r**2 + 2 * s * w / r**2 + s * div / (3 * r)
    force_r = ddw + 2 * dw / r - (L + 2) * w / r**2 + 2 * s * U / r**2 + ddiv / 3
    thermal_lap = ddtheta + 2 * dtheta / r - L * theta / r**2
    tr = transport(ra, a, dissipation, gamma, pr)
    terms = dict(
        continuity=[sigma * (q - rho * theta) / T, rho * div, bg["drho"] * w],
        horizontal_momentum=[sigma * rho * U, s * q / r, -tr["mu"] * force_h],
        radial_momentum=[
            sigma * rho * w,
            dq,
            bg["g"] * q / T,
            -bg["g"] * rho * theta / T,
            -tr["mu"] * force_r,
        ],
        entropy=[
            sigma * (rho * bg["cp"] * theta - q),
            -rho * bg["cp"] * bg["H"] * w,
            -tr["K"] * thermal_lap,
        ],
    )
    errors = {
        key: float(
            np.max(np.abs(sum(items))) / max(sum(np.max(np.abs(item)) for item in items), 1.0e-300)
        )
        for key, items in terms.items()
    }
    return errors


def _growth_pair(n, eta, ell, ra, a, dissipation, gamma, pr):
    A, B = matrices(n, eta, ell, a, dissipation, gamma, ra, pr)
    values, vectors = eig(A, B, check_finite=False)
    indices = np.where(np.isfinite(values))[0]
    indices = indices[np.argsort(values[indices].real)[::-1]]
    z, _ = chebyshev(n)
    m = n + 1
    pressure_basis = chebvander(2 * z, n - 2)
    rejected = []
    for index in indices:
        sigma, reduced_vector = values[index], vectors[:, index]
        residual = _residual(A, B, sigma, reduced_vector)
        vector = np.concatenate((reduced_vector[: 3 * m], pressure_basis @ reduced_vector[3 * m :]))
        errors = _offgrid_residual(n, eta, ell, ra, sigma, vector, a, dissipation, gamma, pr)
        if residual < 1.0e-9 and max(errors.values()) < 5.0e-6:
            break
        rejected.append(
            dict(
                sigma_real=float(sigma.real),
                sigma_imag=float(sigma.imag),
                offgrid_max=max(errors.values()),
                matrix_residual=residual,
            )
        )
    else:
        raise RuntimeError("No resolved physical growth mode; increase Chebyshev degree")
    chi = transport(ra, a, dissipation, gamma, pr)["chi"]
    return (
        sigma,
        vector,
        dict(
            real=float(sigma.real / chi),
            imag=float(sigma.imag / chi),
            sigma_real=float(sigma.real),
            sigma_imag=float(sigma.imag),
            chi=chi,
            residual=residual,
            offgrid_residuals=errors,
            rejected_candidates=rejected,
        ),
    )


def growth(n, eta, ell, ra, a=1.0, dissipation=0.5, gamma=5 / 3, pr=1.0):
    """Leading poloidal growth rate, including compressible acoustic modes."""
    return _growth_pair(n, eta, ell, ra, a, dissipation, gamma, pr)[2]


def eigenmode(n, eta, ell, ra, a=1.0, dissipation=0.5, gamma=5 / 3, pr=1.0, z=None):
    """Physical eigenmode, all fields scaled to max(abs(w))=chi on output z.

    Default output is the ascending Lobatto grid. Supply a 1-D array z in the
    shell for deterministic barycentric interpolation. w,U,Tprime,pprime are
    physical (unseeded) amplitudes; V=U/sqrt(L) and rhoprime are also returned.
    A real stationary branch is required for this real-valued runner export.
    """
    if int(ell) != ell:
        raise ValueError("Physical eigenmodes require integer ell")
    sigma, vector, info = _growth_pair(n, eta, ell, ra, a, dissipation, gamma, pr)
    nodes, _ = chebyshev(n)
    m = len(nodes)
    vector *= np.exp(-1j * np.angle(vector[m : 2 * m][np.argmax(np.abs(vector[m : 2 * m]))]))
    if abs(sigma.imag) > 1.0e-8 or np.max(np.abs(vector.imag)) > 1.0e-7 * np.max(
        np.abs(vector.real)
    ):
        raise RuntimeError("Leading mode is oscillatory; cannot export one real profile")
    samples = nodes[::-1] if z is None else np.asarray(z, dtype=float)
    if samples.ndim != 1 or len(samples) == 0 or np.any(np.abs(samples) > 0.5 + 1.0e-12):
        raise ValueError("z must be a nonempty one-dimensional array inside [-.5,.5]")
    # Exact Lobatto weights avoid randomized weight construction and large-r loss.
    weights = (-1.0) ** np.arange(m)
    weights[[0, -1]] *= 0.5
    interp = BarycentricInterpolator(nodes, vector.real.reshape(4, m), axis=1, wi=weights)
    U, w, temperature, pressure = interp(samples)
    scale = info["chi"] / np.max(np.abs(w))
    U, w, temperature, pressure = [field * scale for field in (U, w, temperature, pressure)]
    bg = background(samples, eta, a, dissipation, gamma)
    return dict(
        z=samples,
        r=bg["r"],
        w=w,
        U=U,
        V=U / np.sqrt(ell * (ell + 1)),
        Tprime=temperature,
        pprime=pressure,
        rhoprime=(pressure - bg["rho"] * temperature) / bg["T"],
        **info,
    )


def export_profile(path, nr, eta, ell, ra, n=48, a=1.0, dissipation=0.5, gamma=5 / 3, pr=1.0):
    """Write exactly nr headerless rows z,w,U,Tpert,ppert on active centers."""
    if int(nr) != nr or nr < 2:
        raise ValueError("nr must be an integer >=2")
    z = (np.arange(nr) + 0.5) / nr - 0.5
    mode = eigenmode(n, eta, ell, ra, a, dissipation, gamma, pr, z=z)
    np.savetxt(
        path,
        np.column_stack([mode[key] for key in ("z", "w", "U", "Tprime", "pprime")]),
        fmt="%.17e",
    )
    return mode


def _background_checks(eta, a=1.0, dissipation=0.5, gamma=5 / 3):
    z, derivative = chebyshev(48)
    b = background(z, eta, a, dissipation, gamma)
    mid = background(0.0, eta, a, dissipation, gamma)
    assert abs(float(mid["T"]) - 1) < 1.0e-14
    assert abs(float(mid["rho"]) - 1) < 1.0e-14
    assert abs(b["T"][-1] - b["T"][0] - a) < 2.0e-12
    integral_error = max(
        abs(
            float(background(x, eta, a, dissipation, gamma)["J"])
            - quad(
                lambda y: 1 / float(background(y, eta, a, dissipation, gamma)["T"]),
                0,
                x,
                epsabs=1.0e-13,
                epsrel=1.0e-13,
            )[0]
        )
        for x in (-0.5, -0.25, 0.25, 0.5)
    )
    hydro = np.max(np.abs(derivative @ b["p"] + b["g"] * b["rho"]))
    conductive = np.max(np.abs(derivative @ b["T"] - b["dT"]))
    assert integral_error < 2.0e-12
    assert hydro < 2.0e-9 and conductive < 2.0e-9
    return dict(
        T_inner=float(b["T"][-1]),
        T_outer=float(b["T"][0]),
        rho_inner=float(b["rho"][-1]),
        rho_outer=float(b["rho"][0]),
        H_inner=float(b["H"][-1]),
        H_outer=float(b["H"][0]),
        integral_error=integral_error,
        hydrostatic_residual=float(hydro),
        temperature_derivative_error=float(conductive),
    )


def _vector_operator_checks():
    """Check curvature/cross terms against Cartesian vector identities.

    For u=grad(r**p Y_ell), div(tau)/mu=(4/3)grad(Delta(r**p Y_ell)).
    ell=p=1 is uniform Cartesian translation (zero force); ell=2,p=4
    also exercises nonzero divergence and all radial/tangential coupling.
    """
    errors = []
    n = 24
    m = n + 1
    z, _ = chebyshev(n)
    r = background(z, 0.5)["r"]
    for ell, power in ((1, 1), (2, 4)):
        s = np.sqrt(ell * (ell + 1))
        U, w = s * r ** (power - 1), power * r ** (power - 1)
        A, _ = matrices(n, 0.5, ell, reduced_pressure=False)
        force = A[m : 3 * m, : 2 * m] @ np.concatenate((U, w))
        coefficient = power * (power + 1) - ell * (ell + 1)
        exact = np.concatenate(
            (
                4 / 3 * s * coefficient * r ** (power - 3),
                4 / 3 * (power - 2) * coefficient * r ** (power - 3),
            )
        )
        interior = np.ones(2 * m, dtype=bool)
        interior[[0, m - 1, m, 2 * m - 1]] = False
        error = float(np.max(np.abs(force[interior] - exact[interior])))
        assert error < 2.0e-8
        errors.append(dict(ell=ell, radial_power=power, max_absolute_error=error))
    return errors


def _profile_validation():
    """Regression for the former weakly damped, oscillatory-pressure artifact."""
    records = []
    centers = (np.arange(128) + 0.5) / 128 - 0.5
    for ra in (820.0, 900.0):
        modes = [eigenmode(n, 0.5, 3, ra, z=centers) for n in (24, 32, 48, 64)]
        chi = modes[-1]["chi"]
        profile_error = {
            key: max(float(np.max(np.abs(mode[key] - modes[-1][key])) / chi) for mode in modes[:-1])
            for key in ("w", "U", "Tprime", "pprime")
        }
        rate_error = max(abs(mode["real"] - modes[-1]["real"]) for mode in modes[:-1])
        assert max(profile_error.values()) < 2.0e-7
        assert rate_error < 2.0e-7
        records.append(
            dict(
                eta=0.5,
                ell=3,
                Ra=ra,
                sample_count=len(centers),
                levels=[
                    dict(N=n, growth=mode["real"], offgrid=mode["offgrid_residuals"])
                    for n, mode in zip((24, 32, 48, 64), modes)
                ],
                max_profile_difference_over_chi=profile_error,
                max_growth_difference=rate_error,
            )
        )
    # Deliberately recover the old equal-degree artifact so the regression
    # establishes why a tiny matrix residual cannot certify a physical mode.
    A, B = matrices(48, 0.5, 3, ra=820, reduced_pressure=False)
    values, vectors = eig(A, B, check_finite=False)
    indices = np.flatnonzero(np.isfinite(values))
    j = indices[np.argmax(values[indices].real)]
    errors = _offgrid_residual(48, 0.5, 3, 820, values[j], vectors[:, j], 1, 0.5, 5 / 3, 1)
    assert max(errors.values()) > 0.5
    bad_rate = float(values[j].real / transport(820)["chi"])
    assert abs(bad_rate - records[0]["levels"][-1]["growth"]) > 0.02
    return dict(
        corrected_profiles=records,
        rejected_equal_degree_example=dict(
            eta=0.5,
            ell=3,
            Ra=820,
            N=48,
            growth=bad_rate,
            matrix_residual=_residual(A, B, values[j], vectors[:, j]),
            offgrid=errors,
        ),
    )


def reference_sweep():
    """Compute, assert consistency, and return the serializable reference table."""
    from planar_compressible_reference import critical as planar_critical

    z, derivative = chebyshev(24)
    assert np.max(np.abs(derivative @ z**3 - 3 * z**2)) < 1.0e-10
    # Explicitly exercise A=0, which a direct background formula divides by.
    _background_checks(0.5, a=0.75, dissipation=0.25)
    vector_checks = _vector_operator_checks()
    profile_validation = _profile_validation()
    planar = planar_critical(48, 1.0, 0.5)
    cases = []
    for eta in ETAS:
        bg_checks = _background_checks(eta)
        levels = [critical(n, eta) for n in (24, 32, 48)]
        best = levels[-1]
        assert len({row["ell"] for row in levels}) == 1
        assert abs(levels[-1]["Ra"] - levels[-2]["Ra"]) < 2.0e-5
        assert abs(levels[-2]["Ra"] - levels[-3]["Ra"]) < 2.0e-5
        rates = [
            dict(Ra=factor * best["Ra"], **growth(48, eta, best["ell"], factor * best["Ra"]))
            for factor in (0.98, 1.0, 1.02)
        ]
        assert rates[0]["real"] < 0 < rates[-1]["real"]
        assert abs(rates[1]["real"]) < 2.0e-6
        assert max(abs(item["imag"]) for item in rates) < 1.0e-7
        growth_levels = [
            dict(N=n, **growth(n, eta, best["ell"], 1.02 * best["Ra"])) for n in (24, 32, 48)
        ]
        assert max(abs(item["real"] - rates[-1]["real"]) for item in growth_levels) < 2.0e-6
        # Include acoustic/thermal eigenvalues at the sampled global angular
        # degrees, to check for a lower oscillatory threshold on that sample.
        rm = geometry(eta)[2]
        degrees = {
            max(1, int(round((np.sqrt(1 + 4 * (k * rm) ** 2) - 1) / 2)))
            for k in best["global_bracket"]["sampled_k"]
        }
        degrees.update(item["ell"] for item in best["integer_candidates"])
        stable_scan = [
            dict(ell=ell, **growth(32, eta, ell, 0.98 * best["Ra"])) for ell in sorted(degrees)
        ]
        assert max(item["real"] for item in stable_scan) < 0
        # An independent neutral pencil residual and physical wall residuals.
        A, B = matrices(48, eta, best["ell"])
        values, vectors = eig(A, B, check_finite=False)
        j = int(np.argmin(np.abs(values - best["Ra"])))
        neutral_residual = _residual(A, B, values[j], vectors[:, j])
        assert neutral_residual < 1.0e-9
        mode = eigenmode(48, eta, best["ell"], 1.02 * best["Ra"])
        _, Dr = chebyshev(48)
        traction = Dr @ mode["U"][::-1] - mode["U"][::-1] / mode["r"][::-1]
        wall_error = (
            max(
                np.max(np.abs(mode["w"][[0, -1]])),
                np.max(np.abs(mode["Tprime"][[0, -1]])),
                np.max(np.abs(traction[[0, -1]])),
            )
            / mode["chi"]
        )
        assert wall_error < 2.0e-7
        center_z = (np.arange(32) + 0.5) / 32 - 0.5
        sample = eigenmode(48, eta, best["ell"], best["Ra"], z=center_z)
        assert abs(np.max(np.abs(sample["w"])) / sample["chi"] - 1) < 1.0e-14
        sample_json = dict(
            Ra=best["Ra"],
            ell=best["ell"],
            nr=32,
            chi=sample["chi"],
            normalization="max(abs(w))=chi on these centers; multiply all fields by runner seed",
            **{
                key: sample[key].tolist()
                for key in ("z", "w", "U", "V", "Tprime", "pprime", "rhoprime")
            },
        )
        row = dict(
            eta=eta,
            levels=levels,
            growth=rates,
            background=bg_checks,
            neutral_residual=neutral_residual,
            physical_wall_residual=wall_error,
            growth_levels=growth_levels,
            stable_angular_scan=dict(N=32, Ra=0.98 * best["Ra"], samples=stable_scan),
            neutral_eigenmode=sample_json,
            Ra=best["Ra"],
            ell=best["ell"],
            k=best["k"],
            planar_relative_difference=(best["Ra"] / planar["Ra"] - 1),
        )
        cases.append(row)
        print(
            f"eta={eta:.4f} ell={best['ell']:5d} Ra={best['Ra']:.9f} k={best['k']:.8f}", flush=True
        )
    # Curvature is O(d/rm); the .9999 shell need not equal the plane to 1e-5.
    assert abs(cases[-1]["planar_relative_difference"]) < 1.0e-4
    assert abs(cases[-1]["k"] - planar["k"]) < 2.0e-4
    tail_errors = [abs(row["planar_relative_difference"]) for row in cases[-4:]]
    assert all(x > y for x, y in zip(tail_errors, tail_errors[1:]))
    return dict(
        convention="d=T_mid=rho_mid=R=1; eta=ri/ro; a=1; D=.5; gamma=5/3; Pr=1; constant inward g=cp*D; constant mu,K",
        rayleigh="Ra=g*(a-D)/(nu_mid*chi_mid), chi_mid=K/cp, nu_mid=mu",
        background_formula="B=a*ri*ro, A=1-B/rm; T=1-B*z/(r*rm); p=exp(-g*J),rho=p/T; J=z/A-B/A^2*log1p(A*z/rm), or J=z+z^2/(2rm) at A=0",
        velocity_convention="u=w*Y*er+U*grad_Omega(Y)/sqrt(ell*(ell+1)); V=U/sqrt(ell*(ell+1))",
        boundary_conditions="w=0; dU/dr-U/r=0; Tpert=0 at ri,ro",
        method="Independent mixed-degree Chebyshev collocation (U,w,T degree n; p degree n-2; interior continuity), neutral Rayleigh and fully compressible growth pencils; sampled global-k bracket plus neighboring integer degrees; independent dense off-grid differential residual",
        sources=[
            "https://doi.org/10.1017/jfm.2017.108",
            "https://arxiv.org/pdf/1607.02193",
            "https://nschaeff.bitbucket.io/shtns/vsh.html",
        ],
        scope="Poloidal ell>=1 stationary onset; m-degenerate. Global search is a numerical bracket audit, not a proof for unbounded ell. Values are independently computed, not quoted spherical results.",
        vector_operator_checks=vector_checks,
        pressure_profile_validation=profile_validation,
        planar=planar,
        cases=cases,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--export-profile", type=Path)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--ell", type=int)
    parser.add_argument("--ra", type=float)
    parser.add_argument("--nr", type=int, default=32)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--dissipation", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=5 / 3)
    parser.add_argument("--pr", type=float, default=1.0)
    args = parser.parse_args()
    if args.export_profile is not None:
        if args.ell is None or args.ra is None:
            parser.error("--export-profile requires --ell and --ra")
        mode = export_profile(
            args.export_profile,
            args.nr,
            args.eta,
            args.ell,
            args.ra,
            args.n,
            args.a,
            args.dissipation,
            args.gamma,
            args.pr,
        )
        print(
            json.dumps(
                dict(
                    path=str(args.export_profile),
                    rows=args.nr,
                    eta=args.eta,
                    ell=args.ell,
                    Ra=args.ra,
                    growth=mode["real"],
                    chi=mode["chi"],
                    residual=mode["residual"],
                ),
                indent=2,
            )
        )
    else:
        result = reference_sweep()
        path = HERE / "spherical_reference_results.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"All reference checks passed; wrote {path}")


if __name__ == "__main__":
    main()
