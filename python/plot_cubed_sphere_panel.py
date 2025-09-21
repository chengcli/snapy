import numpy as np
import matplotlib.pyplot as plt
from cubed_sphere_utils import (
        ab_limits,
        make_poly_patch,
        draw_single_panel,
        draw_panel_seam,
        draw_panel_corner,
        )

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))

    #draw_single_panel(ax, N=8, nghost=3)

    draw_panel_seam(ax, N=8, nghost=3)

    # ghost zone patches
    (a0, b0), (a1, b1) = ab_limits((1, 0), N=8, nghost=3, exterior=True)
    verts_box = [(a0,b0),(a1,b0),(a1,b1),(a0,b1)]
    poly = make_poly_patch(verts_box, edgecolor='red',
                           facecolor=(0.2,0.6,1.0,0.35),
                           linewidth=1.2)
    ax.add_patch(poly)

    #draw_panel_corner(ax, N=8, nghost=3)

    ax.set_aspect('equal')
    ax.set_xlabel("X (orthographic)")
    ax.set_ylabel("Y (orthographic)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    plt.show()
