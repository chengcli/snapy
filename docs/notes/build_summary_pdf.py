"""Build the final eight-page PDF summary from the completed study artifacts.

Run from the repository root: python docs/notes/build_summary_pdf.py
Requires ReportLab, Matplotlib, Pillow, and the DejaVu Sans fonts.
"""

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "snapy-summary-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_spherical_onset import matches_baseline
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from study_archive import read_data

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "cubed_sphere_diffusion_convection_summary.pdf"
WIDTH = A4[0] - 96
INK = colors.HexColor("#16324a")
TEAL = colors.HexColor("#087f8c")
LIGHT = colors.HexColor("#edf4f7")


def main():
    paths = [
        HERE / name
        for name in (
            "cubed_sphere_diffusion_convection.rst",
            "spherical_compressible_theory.inc",
            "spherical_reference_results.json",
            "spherical_onset_results.json",
            "spherical_control_study.json",
            "planar_onset_results.json",
        )
    ]
    reference = json.loads(read_data(paths[2]))
    results = json.loads(read_data(paths[3]))
    controls = json.loads(read_data(paths[4]))
    if (
        results["status"] != "coverage_complete"
        or controls["status"] != "control_coverage_complete"
    ):
        raise ValueError("The final PDF requires completed primary and supplemental studies")

    # A separate print layout keeps axis labels legible at portrait-page width.
    with plt.rc_context({"font.size": 9, "legend.fontsize": 7.5}):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained")
        cases = sorted(reference["cases"], key=lambda c: c["eta"])
        axes[0].plot(
            [c["eta"] for c in cases],
            [c["Ra"] for c in cases],
            "k.-",
            lw=1,
            ms=3,
            label="Spherical reference",
        )
        axes[0].axhline(
            reference["planar"]["Ra"], color="gray", ls="--", lw=1, label="Planar reference"
        )
        for i, nr in enumerate((8, 12, 16)):
            groups = [
                g
                for g in results["groups"]
                if "neutral" in g
                and g["parameters"]["nr"] == nr
                and matches_baseline(g, results["coverage"]["baseline"])
            ]
            eta = [g["parameters"]["eta"] for g in groups]
            values = [g["neutral"]["Ra"] for g in groups]
            errors = [100 * g["neutral"]["relative_error"] for g in groups]
            for ax, y in zip(axes, (values, errors)):
                ax.plot(
                    eta,
                    y,
                    ("o", "s", "^")[i],
                    color=f"C{i}",
                    ms=4,
                    label=f"GPU nr={nr}",
                    linestyle="none",
                )
        axes[0].set(title="Modal onset", ylabel="Rayleigh number")
        axes[0].legend(loc="upper right", framealpha=0.95)
        axes[1].set(title="Error vs spherical reference", ylabel="Relative error (%)")
        axes[1].axhline(0, color="gray", lw=1)
        axes[1].legend(loc="upper left")
        for ax in axes:
            ax.set_xlabel("Radius ratio η")
            ax.grid(alpha=0.2)
        fig.savefig(HERE / "figures/spherical_summary_onset.png", dpi=240)
        plt.close(fig)

    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, file in (
        ("DejaVu", "DejaVuSans.ttf"),
        ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
        ("DejaVu-Italic", "DejaVuSans-Oblique.ttf"),
        ("DejaVuMono", "DejaVuSansMono.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(font_path / file)))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Italic"
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Text", fontName="DejaVu", fontSize=9.6, leading=14, spaceAfter=8))
    styles.add(
        ParagraphStyle(
            "TitleCustom",
            fontName="DejaVu-Bold",
            fontSize=25,
            leading=30,
            textColor=INK,
            spaceAfter=15,
        )
    )
    styles.add(
        ParagraphStyle(
            "Section", fontName="DejaVu-Bold", fontSize=18, leading=23, textColor=INK, spaceAfter=12
        )
    )
    styles.add(
        ParagraphStyle(
            "Sub",
            fontName="DejaVu-Bold",
            fontSize=11,
            leading=15,
            textColor=TEAL,
            spaceBefore=7,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "CaptionCustom",
            fontName="DejaVu",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#4c5965"),
            spaceAfter=9,
        )
    )
    styles.add(ParagraphStyle("Cell", fontName="DejaVu", fontSize=8.2, leading=11))
    styles.add(
        ParagraphStyle(
            "CodeCustom",
            fontName="DejaVuMono",
            fontSize=7.1,
            leading=10,
            backColor=LIGHT,
            borderPadding=7,
            spaceAfter=9,
        )
    )
    styles.add(ParagraphStyle("Kicker", parent=styles["Text"], textColor=TEAL, fontSize=9))
    story = []

    def p(text, style="Text"):
        story.append(Paragraph(text, styles[style]))

    def section(title):
        if story:
            story.append(PageBreak())
        p(title, "Section")

    def table(rows, widths):
        cells = [[Paragraph(str(c), styles["Cell"]) for c in row] for row in rows]
        t = Table(cells, colWidths=[WIDTH * w for w in widths], repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f7f9fa")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
                ]
            )
        )
        story.extend([t, Spacer(1, 9)])

    def figure(name, caption, width=WIDTH):
        path = HERE / "figures" / name
        paths.append(path)
        with PILImage.open(path) as im:
            height = width * im.height / im.width
        story.append(Image(str(path), width=width, height=height))
        p(caption, "CaptionCustom")

    def equation(lines, size=13):
        # Render equations at print resolution; all prose and tables remain searchable.
        fig = plt.figure(figsize=(7, 0.48 * len(lines)))
        for i, line in enumerate(lines):
            fig.text(
                0.5,
                1 - (i + 0.55) / len(lines),
                "$" + line + "$",
                ha="center",
                va="center",
                fontsize=size,
            )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=280, bbox_inches="tight", pad_inches=0.07)
        plt.close(fig)
        buf.seek(0)
        with PILImage.open(buf) as im:
            w, h = im.size
        scale = min(72 / 280, WIDTH / w)
        buf.seek(0)
        story.extend([Image(buf, width=w * scale, height=h * scale), Spacer(1, 8)])

    # 1. Standalone executive summary.
    p("SNAPY  /  FINAL STUDY SUMMARY  /  25 OCTOBER 2026", "Kicker")
    p("Diffusion and convection<br/>on the cubed sphere", "TitleCustom")
    p("Geometry, compressible stability, and GPU verification", "Sub")
    p(
        "Conservative cubed-sphere heat conduction and Newtonian viscosity are implemented. "
        "An independent compressible spherical-harmonic stability calculation provides the "
        "reference for a completed six-panel GPU onset study over radius ratios η = 0.5–0.99."
    )
    table(
        [
            ["<b>Main finding</b>", "<b>Result</b>"],
            [
                "Compressible planar benchmark",
                "GPU Ra = 696.00945; spectral Ra = 696.16667 (0.0226% difference).",
            ],
            [
                "Spherical GPU study",
                f"{len(results['runs'])} analyzed completed runs; eight radius ratios, "
                "at least two grids each; finest-grid errors below 0.80%. Three "
                "additional long-duration runs.",
            ],
            [
                "Continuum thin-shell limit",
                "Spherical Ra = 696.17613 at η = 0.9999, approaching the "
                "matched compressible planar value 696.16667.",
            ],
            [
                "Time integration actually used",
                "Explicit radial and horizontal conduction and viscosity, "
                "with SSPRK3. Optional implicit radial heat conduction is disabled "
                "in the onset runs.",
            ],
            [
                "Remaining limitation",
                "Thin-shell GPU grid error is larger than the curvature correction. "
                "The measurements do not establish a precision continuum extrapolation.",
            ],
        ],
        [0.31, 0.69],
    )
    p("Three different thresholds", "Sub")
    p(
        "The stress-free <b>Boussinesq</b> planar threshold is 657.51136. The "
        "present finite-stratification "
        "<b>compressible</b> planar threshold is 696.16667. The <b>finite-grid "
        "GPU</b> thresholds also "
        "contain spatial and finite-time errors. These quantities are not interchangeable."
    )
    p("Scope and conventions", "Sub")
    p(
        "η = rᵢ/rₒ is a radius ratio, not an absolute shell thickness. Increasing "
        "η makes the shell "
        "thinner relative to its radius. All matched onset comparisons use a "
        "nonrotating ideal gas, "
        "constant inward gravity, constant dynamic viscosity and conductivity, and impermeable, "
        "stress-free, fixed-temperature walls."
    )
    p(
        "This PDF summarizes the completed study. Full derivations, archived data, "
        "exact run commands, "
        "and detailed qualifications are in docs/notes/cubed_sphere_diffusion_convection.rst.",
        "CaptionCustom",
    )

    # 2. Physical operators and geometric distinction.
    section("1  Geometry and conservative diffusion")
    p(
        "On the +X panel, use X = tan α, Y = tan β, and S = 1 + X² + Y². Other "
        "panels are rotations. "
        "The radial direction is orthogonal to the sphere; the two horizontal "
        "coordinate directions "
        "are generally not orthogonal."
    )
    equation(
        [
            r"\mathbf{x}=r(1,X,Y)/\sqrt{S},\quad g_{rr}=1,\quad g_{ra}=0,\quad "
            r"g_{ab}=r^2\gamma_{ab}",
            r"J=\sqrt{\det\gamma}=\frac{(1+X^2)(1+Y^2)}{S^{3/2}},\qquad \sqrt{g}=r^2J",
            r"\gamma^{\alpha\alpha}=\frac{S}{1+X^2},\quad \gamma^{\beta\beta}=\frac{S}{1+Y^2},\quad"
            r"\gamma^{\alpha\beta}=\frac{SXY}{(1+X^2)(1+Y^2)}",
        ],
        12,
    )
    figure(
        "cubed_sphere_geometry.png",
        "Figure 1. Six panels and horizontal basis-angle variation; "
        "the nonorthogonality is geometric and persists under refinement.",
        WIDTH * 0.90,
    )
    p(
        "Apply the coordinate divergence to the physical flux −D∇φ. The full inverse metric, "
        "including its off-diagonal term, gives"
    )
    equation(
        [
            r"\mathcal{L}_D\phi=\frac{1}{r^2}\partial_r(r^2D\partial_r\phi)"
            r"+\frac{1}{r^2J}\partial_a(JD\gamma^{ab}\partial_b\phi)."
        ],
        13,
    )
    p(
        "For heat conduction, D = K and φ = T; add this divergence to the compressible energy "
        "equation. A temperature diffusion equation alone omits pressure work and viscous heating. "
        "For constant K, the exact conductive state is T = A + B/r and 4πr²qᵣ is constant."
    )
    p("Viscosity requires the tensor stress", "Sub")
    equation(
        [
            r"\boldsymbol{\tau}=\mu\left[\nabla\mathbf{u}+(\nabla\mathbf{u})^T"
            r"-\frac{2}{3}\mathbf{I}\,\nabla\!\cdot\mathbf{u}\right]."
        ],
        13,
    )
    p(
        "Reconstruct Cartesian gradients and face tractions, then project into the stored momentum "
        "basis. Independent scalar Laplacians of panel velocity components omit basis derivatives. "
        "Use one common integrated flux across each seam, with opposite signs in its two cells."
    )

    # 3. Boussinesq derivation, explicitly distinct from the compressible study.
    section("2  Thin-shell Boussinesq onset")
    p(
        "Let d = rₒ − rᵢ, rₘ = (rᵢ + rₒ)/2, ε = d/rₘ ≪ 1, and ζ = (r − rᵢ)/d. "
        "Scale time by d²/χ, velocity by χ/d, and temperature perturbations by the imposed drop. "
        "The leading-order linear equations are"
    )
    equation(
        [
            r"Pr^{-1}\partial_t\mathbf{u}=-\nabla p+\nabla^2\mathbf{u}+Ra\,\theta\,\mathbf{e}_r,"
            r"\qquad \nabla\!\cdot\mathbf{u}=0,",
            r"\partial_t\theta=w+\nabla^2\theta,\qquad"
            r"Ra=\frac{g\alpha_T\Delta T d^3}{\nu\chi},\quad Pr=\frac{\nu}{\chi}.",
        ],
        12,
    )
    p(
        "Expand the angular dependence in Yℓm, for which ΔΩYℓm = −ℓ(ℓ+1)Yℓm. "
        "The dimensionless horizontal wavenumber is aℓ² = ε²ℓ(ℓ+1). At stationary onset, "
        "eliminating pressure gives the coupled vertical equations"
    )
    equation(
        [
            r"(\partial_\zeta^2-a_\ell^2)^2w=Ra\,a_\ell^2\theta,\qquad"
            r"(\partial_\zeta^2-a_\ell^2)\theta=-w.",
        ],
        12,
    )
    p(
        "Impermeable stress-free walls give w = ∂²ζw = 0, and fixed temperatures give θ = 0. "
        "Choose w, θ proportional to sin(nπζ); each Laplacian contributes −(n²π² + aℓ²). Hence"
    )
    equation(
        [
            r"Ra_{\ell n}=\frac{(n^2\pi^2+a_\ell^2)^3}{a_\ell^2},\qquad"
            r"Ra_c=\min_{\ell\geq1}Ra_{\ell 1}.",
            r"a_c^2=\pi^2/2,\qquad Ra_c\longrightarrow27\pi^4/4=657.5113645",
        ],
        15,
    )
    p(
        "The second line follows by minimizing over a continuous horizontal wavenumber as ε → 0. "
        "At finite ε, integer harmonic degrees produce a discrete minimum. This is a leading-order "
        "thin-shell result, not the exact finite-thickness spherical threshold. No-slip walls have "
        "a different critical value."
    )
    p("How this relates to the compressible calculations", "Sub")
    table(
        [
            ["<b>Limit</b>", "<b>Threshold and interpretation</b>"],
            [
                "Weak stratification and thin shell",
                "With a = 10⁻⁴ and dissipation number 5 × 10⁻⁵, "
                "the compressible spherical solver gives 657.51136777 at η = 0.9999; "
                "the analytic Boussinesq value is 657.51136448.",
            ],
            [
                "Thin shell at finite stratification",
                "With a = 1 and dissipation number 0.5, "
                "the target remains compressible: 696.16667. Geometric flattening "
                "does not eliminate "
                "density stratification or pressure fluctuations.",
            ],
        ],
        [0.32, 0.68],
    )

    # 4. Independent compressible reference construction.
    section("3  Independent compressible reference")
    p(
        "Use midpoint units d = Tₘ = ρₘ = R<sub>gas</sub> = 1. Set a = ΔT/Tₘ = 1, "
        "D = gd/(cₚTₘ) = 0.5 (dissipation number), γ = 5/3, cₚ = 2.5, g = 1.25, and Pr = 1. "
        "The Rayleigh convention uses the integrated superadiabatic drop and "
        "midpoint diffusivities."
    )
    equation(
        [
            r"T_b=A+B/r,\quad B=a r_i r_o,\quad A=1-B/r_m,\quad p_b'=-g\rho_b,\quad "
            r"p_b=\rho_b T_b,",
            r"Ra_{sa}=\frac{g(a-\mathcal{D})}{\nu_m\chi_m},\qquad"
            r"\nu_m=\mu/\rho_m,\quad\chi_m=K/(\rho_m c_p).",
        ],
        11.8,
    )
    p(
        "This background preserves Tₘ = ρₘ = 1 and the wall-temperature difference at every radius "
        "ratio. Its wall temperatures are 1 + a/(1+η) and 1 − aη/(1+η). A linear radial "
        "temperature or fixed planar wall temperatures would define a different spherical problem."
    )
    p("Separate angular dependence analytically", "Sub")
    equation(
        [
            r"\mathbf{u}=wY_{\ell m}\mathbf{e}_r+U\nabla_\Omega Y_{\ell m}/s,\qquad "
            r"s=\sqrt{\ell(\ell+1)},",
            r"\mathcal{S}=\partial_r^2+2\partial_r/r-s^2/r^2,\quad"
            r"\delta=w'+2w/r-sU/r,\quad H=-T_b'-\mathcal{D}.",
        ],
        12,
    )
    p(
        "With temperature and pressure amplitudes θ and q, the full growth problem retains "
        "density and pressure time derivatives:"
    )
    equation(
        [
            r"\sigma(q-\rho_b\theta)/T_b=-\rho_b\delta-\rho_b'w,",
            r"\sigma\rho_b U=-s q/r+\mu F_h,",
            r"\sigma\rho_b w=-q'-gq/T_b+g\rho_b\theta/T_b+\mu F_r,",
            r"\sigma(\rho_b c_p\theta-q)=\rho_b c_p H w+K\mathcal{S}\theta.",
        ],
        12,
    )
    p(
        "Fₕ = <i>S</i>U + 2sw/r² + sδ/(3r), and Fᵣ = <i>S</i>w − 2w/r² + 2sU/r² + δ′/3. "
        "At both walls impose w = 0, U′ − U/r = 0, and θ = 0. Setting σ = 0 gives "
        "a generalized Rayleigh eigenproblem; the growth problem separately checks the sign change."
    )
    p("Numerical verification of the reference", "Sub")
    p(
        "Chebyshev degree n is used for velocities and temperature and degree n−2 for pressure. "
        "This removes an unresolved equal-degree pressure branch. Degrees 24, 32, and 48 agree "
        "in critical Ra to less than 2 × 10⁻⁹. Independent off-grid differential residuals, "
        "neighboring growth rates, and integer-degree searches are checked. The angular search "
        "is audited numerically; it is not a theorem over an unbounded spectrum."
    )

    # 5. Implementation and what was actually run.
    section("4  Numerical scheme and GPU protocol")
    table(
        [
            ["<b>Part</b>", "<b>Implementation / rationale</b>"],
            [
                "Horizontal diffusion",
                "Conservative finite-volume face fluxes with the full metric, "
                "including cross derivatives. Common seam fluxes preserve global cancellation.",
            ],
            [
                "Radial diffusion",
                "Spherical face-area and cell-volume weights. Radial lines contain "
                "no angular cross derivative and can be solved independently for heat conduction.",
            ],
            [
                "Optional implicit heat solver",
                "SDIRK(2,2), with Strang half-steps around a hydro step. "
                "Dry ideal gas only. Horizontal heat and all viscous terms remain explicit.",
            ],
            [
                "Actual onset integration",
                "Implicit radial heat solver OFF. Explicit conduction and "
                "viscosity in all directions, SSPRK3, WENO5 reconstruction, LMARS fluxes, float64.",
            ],
            [
                "Timestep",
                "CFL 0.8; diffusion_step_scale = 8 relaxes the conservative diffusion guard. "
                "The acoustic cap remains. This option does not create an implicit update.",
            ],
        ],
        [0.29, 0.71],
    )
    p(
        "The operator is an architecture-based choice, not the winner of an "
        "exhaustive scheme ranking. "
        "Implicit radial conduction can relieve a restrictive radial heat timestep; it does not "
        "remove horizontal, viscous, or acoustic restrictions. It is not needed to define the "
        "stationary threshold and was excluded from these onset comparisons."
    )
    p("Initialization and measurement", "Sub")
    p(
        "All six panels evolve the fully compressible production equations. Initialize the exact "
        "conductive temperature, apply discrete hydrostatic balance, normalize midpoint density, "
        "and seed all four physical perturbations from an independent compressible eigenmode "
        "with amplitude 10⁻⁶. The evolving state is neither filtered nor projected "
        "back onto a harmonic."
    )
    equation(
        [
            r"f=\sin[\pi(r-r_i)/d]Y_{\ell0},\qquad"
            r"A(t)=\frac{\sum V f u_r}{\sum V f^2},\qquad \tau=t\chi_m/d^2.",
        ],
        12,
    )
    p(
        "Fit log|A| versus τ in 0.05–0.10, 0.10–0.15, and 0.15–0.20. Three Rayleigh numbers "
        "(0.94, 1, and 1.06 times the reference) give a growth/decay bracket; interpolate its zero "
        "with a quadratic. Report the last window and its shift from the preceding window."
    )
    p("Cost and distributed execution", "Sub")
    p(
        "A four-point sparse stencil replaces dense polynomial multiplication; "
        "boundary-only traces "
        "and streamed seam buffers reduce memory. The largest run has 6 × 504² × 12 = 18,289,152 "
        "active cells on two GPUs. On this host, a matched ten-step transport probe took 170.89 s "
        "with CUDA IPC and 5.27 s with copy staging, with identical recorded diagnostics. "
        "Concurrent jobs were active and the cause was not profiled, so this is not a general "
        "scaling result. Remaining distributed runs used UCX_TLS=self,tcp,cuda_copy."
    )

    # 6. Final numerical results, read directly from the archive.
    section("5  Final spherical GPU results")
    p(
        "All eight required radius ratios have at least two completed grids. The table selects the "
        "finest primary grid at each ratio. N is the number of angular cells per panel direction; "
        "nᵣ is the number of radial cells. Each threshold is a measured seeded-mode neutral point."
    )
    rows = [
        [
            "<b>η</b>",
            "<b>ℓ</b>",
            "<b>N / nᵣ</b>",
            "<b>Spectral Ra</b>",
            "<b>GPU Ra</b>",
            "<b>Error</b>",
        ]
    ]
    for eta in results["coverage"]["required_ratios"]:
        groups = [
            g
            for g in results["groups"]
            if "neutral" in g
            and g["parameters"]["eta"] == eta
            and matches_baseline(g, results["coverage"]["baseline"])
        ]
        g = max(groups, key=lambda x: (x["parameters"]["nr"], x["parameters"]["N"]))
        a = g["parameters"]
        rows.append(
            [
                f"{eta:g}",
                str(a["ell"]),
                f"{a['N']} / {a['nr']}",
                f"{g['reference']['modal_Ra']:.5f}",
                f"{g['neutral']['Ra']:.5f}",
                f"{100 * g['neutral']['relative_error']:+.4f}%",
            ]
        )
    table(rows, [0.09, 0.07, 0.17, 0.23, 0.23, 0.21])
    figure(
        "spherical_summary_onset.png",
        "Figure 2. All primary grid families and errors relative "
        "to the same-degree spectral threshold. Unconnected points retain "
        "the distinction between different grids at the same radius ratio.",
    )
    p("Separate horizontal and radial refinement", "Sub")
    rows = [["<b>η / direction</b>", "<b>N / nᵣ change</b>", "<b>Ra change</b>"]]
    for c in results["refinement_comparisons"]:
        if c.get("refined_direction") not in ("angular", "radial"):
            continue
        a, b = c["coarse_grid"], c["fine_grid"]
        rows.append(
            [
                f"{c['parameters']['eta']:g} / {c['refined_direction']}",
                f"{a['N']}/{a['nr']} → {b['N']}/{b['nr']}",
                f"{c['coarse_neutral_Ra']:.3f} → {c['fine_neutral_Ra']:.3f}",
            ]
        )
    table(rows, [0.27, 0.32, 0.41])
    p(
        "At η = 0.9 the radial-only change is +3.470 Ra, versus +1.147 for the "
        "angular-only change. "
        "At η = 0.99, angular refinement removes the original coarse-grid upward turn. "
        "These directional changes do not establish an asymptotic convergence order.",
        "CaptionCustom",
    )

    # 7. Honest interpretation of the thin-shell limit.
    section("6  Thin-shell limit and remaining error")
    figure(
        "spherical_thin_shell_limit.png",
        "Figure 3. Differences from the compressible planar threshold. "
        "Vertical bars show the last fitting-window shift, not a confidence interval or a bound "
        "on spatial error.",
        WIDTH * 0.95,
    )
    p("The independent reference approaches the plane", "Sub")
    p(
        f"The spherical reference decreases from 865.98306 at η = 0.5 to 697.13666 at η = 0.99. "
        f"At η = 0.9999 it is 696.17613, only 0.00946 above the independently calculated "
        f"planar value {reference['planar']['Ra']:.5f}. Curvature terms vanish and permitted "
        "horizontal wavenumbers become dense in this limit."
    )
    p("The GPU curve still contains spatial error", "Sub")
    p(
        "At η = 0.98, the continuum curvature correction is only 1.99 Ra, whereas the "
        "N = 336, nᵣ = 12 GPU value lies 5.48 Ra below its spherical reference. At η = 0.99, "
        "N = 504, nᵣ = 12 gives 691.57820, versus 697.13666 from the spherical calculation. "
        "Its last-window shift is 0.23324 Ra; that does not bound the roughly 5.56 "
        "Ra spatial discrepancy."
    )
    p(
        "Scaling N with harmonic degree maintains roughly fixed cells per wavelength. Keeping nᵣ "
        "fixed also preserves radial discretization error. Therefore decreasing relative shell "
        "thickness alone does not force a finite-grid calculation to the continuum planar value."
    )
    p("What is established", "Sub")
    p(
        "The continuum reference has the correct compressible planar and weak-stratification "
        "Boussinesq limits. The GPU study reproduces the spherical thresholds within 0.80% on its "
        "finest primary grids. Precision numerical verification of the small curvature correction "
        "would require further spatial convergence; the η = 0.9 three-grid sequence does not "
        "demonstrate a positive asymptotic order."
    )

    # 8. Controls, reproducibility, references.
    section("7  Controls, validation, and reproducibility")
    figure(
        "spherical_control_study.png",
        "Figure 4. Longer fitting windows: modal growth rates and neutral "
        "thresholds. Window extents are not uncertainty intervals.",
        WIDTH * 0.94,
    )
    table(
        [
            ["<b>Control</b>", "<b>Evidence and implication</b>"],
            [
                "Duration",
                "At η = 0.95, extending τ = 0.2 to 0.6 changes Ra from 696.55594 "
                "to 696.58401. The spatial offset remains about −0.72%.",
            ],
            [
                "Mode / orientation",
                "At η = 0.9, degree 21 remains preferred among 20, 21, 22. "
                "A diagonal seed shifts Ra by +0.65755; this is not an exhaustive "
                "discrete-mode search.",
            ],
            [
                "Background / precision",
                "Zero-seed radial RMS velocity ≤ 1.23 × 10⁻¹². Float32 "
                "near-neutral rates show bias and are excluded from the primary study.",
            ],
            [
                "Validation",
                "50 Python regression tests passed. Native single-/multi-process "
                "cubed-sphere regressions and the full six-panel two-GPU "
                "copy-transport test passed. "
                "The final analysis has no incomplete or invalid runs.",
            ],
        ],
        [0.24, 0.76],
    )
    p("Reproduce the summary and verify the archived study", "Sub")
    p(
        "python docs/notes/build_summary_pdf.py<br/>"
        "python docs/notes/study_archive.py extract<br/>"
        "OPENBLAS_NUM_THREADS=1 python docs/notes/analyze_spherical_onset.py<br/>"
        "&nbsp;&nbsp;--baseline-step-scale 8<br/>"
        "OPENBLAS_NUM_THREADS=1 python -m unittest discover -s docs/notes<br/>"
        "&nbsp;&nbsp;-p 'test_spherical_*.py'",
        "CodeCustom",
    )
    p(
        "Wrapped command lines continue onto the indented line. Raw samples and JSON are in "
        "docs/notes/cubed_sphere_study_data.tar.gz; extract before analysis. Figures and full run "
        "commands remain under docs/notes. Early "
        "samples have incomplete "
        "binary-hash provenance; the final large grids and directional controls share the recorded "
        "final executable and library. Distributed ranks must use matching libraries because the "
        "seam wire layout changed; no public C++ layout or external-repository change is required.",
        "CaptionCustom",
    )
    p("Sources", "Sub")
    p(
        "[1] Alboussière &amp; Ricard (2017), <i>Rayleigh–Bénard stability and the validity of "
        "quasi-Boussinesq or quasi-anelastic liquid approximations</i>, JFM 817, 264–305. "
        '<link href="https://doi.org/10.1017/jfm.2017.108" '
        'color="#087f8c">doi:10.1017/jfm.2017.108</link>. '
        "Supports the compressible planar formulation; spherical values are "
        "derived in this study.<br/>"
        "[2] Full derivation: cubed_sphere_diffusion_convection.rst and "
        "spherical_compressible_theory.inc.<br/>"
        "[3] Numerical evidence: spherical_reference_results.json, spherical_onset_results.json, "
        "spherical_control_study.json, and planar_onset_results.json.",
        "CaptionCustom",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(0.6)
        canvas.line(48, 41, A4[0] - 48, 41)
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(INK)
        canvas.drawString(48, 28, "SNAPY  •  Cubed-sphere diffusion and convection  •  2026-10-25")
        canvas.drawRightString(A4[0] - 48, 28, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=48,
        rightMargin=48,
        topMargin=43,
        bottomMargin=54,
        title="Snapy: cubed-sphere diffusion and convection — final summary",
        author="Snapy technical study",
        subject="Formulation, compressible onset, and GPU verification",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    paths.append(Path(__file__))
    manifest = {
        "pdf": OUTPUT.name,
        "study_date": "2026-10-25",
        "input_sha256": {
            str(p.relative_to(HERE)): hashlib.sha256(read_data(p)).hexdigest()
            for p in dict.fromkeys(paths)
        },
        "pdf_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    (HERE / "cubed_sphere_summary_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
