import numpy as np
import matplotlib.pyplot as plt
from cubed_sphere_utils import draw_single_panel

def draw_panel_seam(ax, N=8, nghost=3):
    # Use view along bisector of +X and +Y directions, i.e. (1,1,0).
    view_dir = np.array([1.0,1.0,0.0])

    ax.set_aspect('equal')

    draw_single_panel(ax, "+X", N=N, nghost=nghost,
                      view_dir=view_dir, color='C0')
    draw_single_panel(ax, "+Y", N=N, nghost=nghost,
                      view_dir=view_dir, color='C1')

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))

    draw_single_panel(ax, view_dir=[1,1,0])
    #draw_panel_seam(ax)

    ax.set_aspect('equal')
    ax.set_xlabel("X (orthographic)")
    ax.set_ylabel("Y (orthographic)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    plt.show()
