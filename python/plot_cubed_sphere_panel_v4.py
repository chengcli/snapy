# Third plot: center the three-face corner (+X, +Y, +Z) by viewing along the space diagonal (1,1,1).
# Overlays +X, +Y, and +Z panels with ghost zones using distinct linestyles.
import numpy as np
import matplotlib.pyplot as plt

def normalize(v):
    return v / np.linalg.norm(v)

def gnomonic_equiangular_to_xyz(alpha, beta, face="+X"):
    t = np.tan(alpha)
    u = np.tan(beta)
    if face == "+X":
        X, Y, Z = np.ones_like(t), t, u
    elif face == "-X":
        X, Y, Z = -np.ones_like(t), -t, u
    elif face == "+Y":
        X, Y, Z = -t, np.ones_like(t), u
    elif face == "-Y":
        X, Y, Z = t, -np.ones_like(t), u
    elif face == "+Z":
        X, Y, Z = -t, u, np.ones_like(t)
    elif face == "-Z":
        X, Y, Z = -t, -u, -np.ones_like(t)
    else:
        raise ValueError("Invalid face")
    inv = 1.0 / np.sqrt(X*X + Y*Y + Z*Z)
    return X*inv, Y*inv, Z*inv

def project_xyz(xyz, view_dir=np.array([1,1,1])):
    x, y, z = xyz
    V = normalize(np.array(view_dir, dtype=float))
    up_guess = np.array([0,0,1.0]) if abs(V[2]) < 0.9 else np.array([0,1.0,0])
    e1 = normalize(np.cross(up_guess, V))
    e2 = np.cross(V, e1)
    U = x*e1[0] + y*e1[1] + z*e1[2]
    W = x*e2[0] + y*e2[1] + z*e2[2]
    return U, W

def draw_face_general(ax, face, N=16, nghost=3, n_pts=700,
                      view_dir=np.array([1,1,1]),
                      interior_style='-', ghost_style='--', boundary_lw=1.6, alpha_grid=1.0):
    dtheta = (np.pi/2) / N
    halfN = N // 2
    idx = np.arange(-halfN - nghost, halfN + nghost + 1)
    lines = idx * dtheta
    s = np.linspace(-np.pi/4 - nghost*dtheta, np.pi/4 + nghost*dtheta, n_pts)

    def is_interior(i):
        return (-halfN) <= i <= (halfN)

    # alpha=const
    for i, a in zip(idx, lines):
        xyz = gnomonic_equiangular_to_xyz(np.full_like(s, a), s, face=face)
        u, v = project_xyz(xyz, view_dir=view_dir)
        ax.plot(u, v, linestyle=(interior_style if is_interior(i) else ghost_style),
                linewidth=0.9, alpha=alpha_grid)
    # beta=const
    for j, b in zip(idx, lines):
        xyz = gnomonic_equiangular_to_xyz(s, np.full_like(s, b), face=face)
        u, v = project_xyz(xyz, view_dir=view_dir)
        ax.plot(u, v, linestyle=(interior_style if is_interior(j) else ghost_style),
                linewidth=0.9, alpha=alpha_grid)

    # boundaries
    for a in [-np.pi/4, np.pi/4]:
        xyz = gnomonic_equiangular_to_xyz(np.full_like(s, a), s, face=face)
        u, v = project_xyz(xyz, view_dir=view_dir)
        ax.plot(u, v, linewidth=boundary_lw)
    for b in [-np.pi/4, np.pi/4]:
        xyz = gnomonic_equiangular_to_xyz(s, np.full_like(s, b), face=face)
        u, v = project_xyz(xyz, view_dir=view_dir)
        ax.plot(u, v, linewidth=boundary_lw)

# View along the cube-space diagonal to center the +X/+Y/+Z corner
view_dir = np.array([1.0, 1.0, 1.0])

fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect('equal')

# Draw three panels meeting at the positive corner
draw_face_general(ax, "+X", N=16, nghost=3, view_dir=view_dir,
                  interior_style='-', ghost_style='--', boundary_lw=1.8, alpha_grid=1.0)
draw_face_general(ax, "+Y", N=16, nghost=3, view_dir=view_dir,
                  interior_style=':', ghost_style='-.', boundary_lw=1.5, alpha_grid=0.95)
draw_face_general(ax, "+Z", N=16, nghost=3, view_dir=view_dir,
                  interior_style='-.', ghost_style='--', boundary_lw=1.5, alpha_grid=0.95)

ax.set_title("Three-Panel Corner View: +X, +Y, +Z\nView rotated to (1,1,1) so the corner is centered")
plt.show()
