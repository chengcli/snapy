#!/usr/bin/env python
"""Summary figure: Straka theta' and Bryan-Fritsch theta_e' per rho_ref form.

  python plot_t3.py <rundir> <straka dx> <bryan dx> out.png
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_t3 import frames  # noqa: E402

FORMS = ["smooth5", "isentrope", "none"]


def main():
    rundir, sdx, bdx, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for j, form in enumerate(FORMS):
        fr = frames(os.path.join(rundir, f"straka_bubble_dx{sdx}_{form}_gpu"))
        v = fr[-1]
        x, z = v["x2"] / 1e3, v["x1"] / 1e3
        thp = v["theta"] - 300.0
        ax = axes[0, j]
        cs = ax.contourf(x, z, thp, levels=np.arange(-10.5, 0.51, 1.0),
                         cmap="Blues_r", extend="both")
        ax.contour(x, z, thp, levels=[-1.0], colors="k", linewidths=0.8)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 5)
        ax.set_title(f"Straka {sdx} m, {form}: theta' min {thp.min():.2f} K")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("z [km]")
    fig.colorbar(cs, ax=axes[0, :], label="theta' [K]", shrink=0.9)

    for j, form in enumerate(FORMS):
        fr = frames(os.path.join(rundir, f"bryan_bubble_dx{bdx}_{form}_gpu"))
        v0, v = fr[0], fr[-1]
        x, z = v["x2"] / 1e3, v["x1"] / 1e3
        tep = v["theta_e"] - v0["theta_e"][:, :1]
        ax = axes[1, j]
        cs = ax.contourf(x, z, tep, levels=np.arange(-0.5, 4.76, 0.25),
                         cmap="RdBu_r", extend="both")
        ax.contour(x, z, tep, levels=[1.0], colors="k", linewidths=0.8)
        ax.set_title(f"BF02 {bdx} m, {form}: theta_e' "
                     f"{tep.max():.2f} / {tep.min():.3f} K")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("z [km]")
    fig.colorbar(cs, ax=axes[1, :], label="theta_e' [K]", shrink=0.9)
    fig.suptitle("#250 T3: Straka (t = 900 s, published theta'min -9.77 K) and "
                 "Bryan & Fritsch moist bubble (t = 1000 s, published "
                 "theta_e' 4.10 / -0.31 K)")
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
