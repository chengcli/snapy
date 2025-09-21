# Rotate the perspective so that the overlayed shared edge (+X and +Y) appears centered in projection.
# For the +X/+Y neighbor pair, the shared edge is along direction (0,1,0) to (0,1,±1)/√2 (the +Y axis region).
# A natural way: view along +Z axis so that the edge appears centered vertically, or along +45° bisector between +X and +Y.
#
# We'll implement a general projection with arbitrary view direction vector, not limited to cube axes.

import numpy as np
import matplotlib.pyplot as plt
from plot_cubed_sphere_panel_v2 import gnomonic_equiangular_to_xyz

def normalize(v):
    return v / np.linalg.norm(v)

def project_xyz(xyz, view_dir=np.array([1,1,0])):
    """
    Project 3D (x,y,z) onto a plane perpendicular to view_dir, using orthographic projection.
    Returns 2D coordinates (u,v).
    """
    x, y, z = xyz
    V = normalize(np.array(view_dir))
    # Construct orthonormal basis (e1,e2) spanning plane perpendicular to V
    # Choose arbitrary "up" vector not parallel to V
    up_guess = np.array([0,0,1]) if abs(V[2])<0.9 else np.array([0,1,0])
    e1 = normalize(np.cross(up_guess, V))
    e2 = np.cross(V, e1)
    # Project coordinates
    U = x*e1[0] + y*e1[1] + z*e1[2]
    W = x*e2[0] + y*e2[1] + z*e2[2]
    return U, W

def draw_face_general(ax, face, N=16, nghost=3, n_pts=700,
                      view_dir=np.array([1,1,0]),
                      interior_style='-', ghost_style='--', boundary_lw=1.6, alpha_grid=1.0):
    dtheta = (np.pi / 2) / N
    halfN = N // 2
    idx = np.arange(-halfN - nghost, halfN + nghost + 1)
    lines = idx * dtheta
    s = np.linspace(-np.pi/4 - nghost*dtheta, np.pi/4 + nghost*dtheta, n_pts)

    def is_interior(i):
        return (-halfN) <= i <= (halfN)

    # alpha=const
    for i, a in zip(idx, lines):
        xyz = gnomonic_equiangular_to_xyz(np.full_like(s, a), s, face=face)
        u,v = project_xyz(xyz, view_dir=view_dir)
        ax.plot(u, v, linestyle=interior_style if is_interior(i) else ghost_style,
                linewidth=0.9, alpha=alpha_grid)
    # beta=const
    for j, b in zip(idx, lines):
        xyz = gnomonic_equiangular_to_xyz(s, np.full_like(s, b), face=face)
        u,v = project_xyz(xyz, view_dir=view_dir)
        ax.plot(u, v, linestyle=interior_style if is_interior(j) else ghost_style,
                linewidth=0.9, alpha=alpha_grid)

    # boundaries bold
    for a in [-np.pi/4, np.pi/4]:
        xyz = gnomonic_equiangular_to_xyz(np.full_like(s, a), s, face=face)
        u,v = project_xyz(xyz, view_dir=view_dir)
        ax.plot(u, v, linewidth=boundary_lw)
    for b in [-np.pi/4, np.pi/4]:
        xyz = gnomonic_equiangular_to_xyz(s, np.full_like(s, b), face=face)
        u,v = project_xyz(xyz, view_dir=view_dir)
        ax.plot(u, v, linewidth=boundary_lw)

# Use view along bisector of +X and +Y directions, i.e. (1,1,0).
view_dir = np.array([1.0,1.0,0.0])

fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect('equal')

draw_face_general(ax, "+X", N=16, nghost=3, view_dir=view_dir,
                  interior_style='-', ghost_style='--', boundary_lw=1.8, alpha_grid=1.0)
draw_face_general(ax, "+Y", N=16, nghost=3, view_dir=view_dir,
                  interior_style=':', ghost_style='-.', boundary_lw=1.5, alpha_grid=0.95)

ax.set_title("Overlay of +X and +Y Panels\nView rotated to bisector direction (1,1,0) so shared edge is centered")
plt.show()
