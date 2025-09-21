import matplotlib.pyplot as plt
from cubed_sphere_utils import plot_single_panel_grid

# Example: 8 interior cells per direction, 3 ghost cells beyond each edge.
fig, ax = plt.subplots(figsize=(8, 8))

# plot visible hemisphere
#theta = np.linspace(0, 2*np.pi, 720)
#ax.plot(np.cos(theta), np.sin(theta), linewidth=1, alpha=0.5)

# all grid lines including ghosts
plot_single_panel_grid(ax, face="+X", N=8, nghost=3, n_pts=800)

# interior grid lines only
plot_single_panel_grid(ax, face="+X", N=8, nghost=0, n_pts=800,
                       linestyle='-', linewidth=1.2,
                       facecolor='C0', color='C0')

ax.set_aspect('equal')
ax.set_xlabel("X (orthographic)")
ax.set_ylabel("Y (orthographic)")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

plt.show()
