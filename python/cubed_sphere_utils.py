import numpy as np

def gnomonic_equiangular_to_xyz(alpha, beta, face="+X"):
    """
    Map equiangular gnomonic coordinates (alpha, beta) in radians to
    Cartesian coordinates (x, y, z) on the unit sphere for a given cube face.

    (alpha, beta) are the equiangular angles from the face center:
        t = tan(alpha), u = tan(beta)
    Faces: "+X", "-X", "+Y", "-Y", "+Z", "-Z"
    """
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
        raise ValueError("Invalid face specifier")

    inv_norm = 1.0 / np.sqrt(X*X + Y*Y + Z*Z)
    return X*inv_norm, Y*inv_norm, Z*inv_norm

def orthographic_project(face_xyz, view_axis="+X"):
    """Orthographic projection onto the plane normal to view_axis."""
    x, y, z = face_xyz
    if view_axis in ("+X", "-X"):
        return y, z
    if view_axis in ("+Y", "-Y"):
        return x, z
    if view_axis in ("+Z", "-Z"):
        return x, y
    raise ValueError("Invalid view axis")

def plot_single_panel_grid(ax, face="+X", N=8, nghost=3, n_pts=600,
                           view_dir = np.array([1,1,0]),
                           color='C0', linestyle='--', linewidth=0.8,
                           facecolor='none'):
    """
    Plot an equiangular gnomonic grid for a single cubed-sphere panel with ghost zones.
      - N: number of interior cells per direction (there are N+1 interior grid lines).
      - nghost: number of extra ghost cells beyond each edge (drawn dashed).
    """
    # Uniform grid in equiangular coordinates; panel spans [-pi/4, +pi/4].
    dtheta = (np.pi / 2) / N  # cell size in angle

    # Grid-line indices: i = -N/2 ... N/2 for interior; extend by nghost
    halfN = N // 2
    idx = np.arange(-halfN - nghost, halfN + nghost + 1)
    alphas = idx * dtheta  # positions of grid lines
    centers = 0.5 * (alphas[1:] + alphas[:-1])  # cell centers

    # Limit plotting domain to a slightly larger band so dashed ghost lines are visible
    s = np.linspace(-np.pi/4 - nghost*dtheta, np.pi/4 + nghost*dtheta, n_pts)

    # Helper to check if a line index is interior or ghost
    def is_interior(i):
        return (-halfN) <= i <= (halfN)

    # alpha = const lines
    for i, alpha in zip(idx, alphas):
        xyz = gnomonic_equiangular_to_xyz(np.full_like(s, alpha), s, face=face)
        u, v = orthographic_project(xyz, view_axis=face)
        #ax.plot(u, v, linewidth=0.9, linestyle='-' if is_interior(i) else '--')
        ax.plot(u, v, linewidth=linewidth, linestyle=linestyle, color=color)

    # beta = const lines
    for j, beta in zip(idx, alphas):
        xyz = gnomonic_equiangular_to_xyz(s, np.full_like(s, beta), face=face)
        u, v = orthographic_project(xyz, view_axis=face)
        #ax.plot(u, v, linewidth=0.9, linestyle='-' if is_interior(j) else '--')
        ax.plot(u, v, linewidth=linewidth, linestyle=linestyle, color=color)

    # cell centers (solid dots)
    center_a, center_b = np.meshgrid(centers, centers)
    xyz = gnomonic_equiangular_to_xyz(center_a, center_b, face=face)
    u, v = orthographic_project(xyz, view_axis=face)
    ax.scatter(u, v, s=10, facecolors=facecolor,
               edgecolors=color, zorder=3)

    # Interior panel boundary (bold): |alpha|=pi/4 and |beta|=pi/4
    for alpha in [-np.pi/4, np.pi/4]:
        xyz = gnomonic_equiangular_to_xyz(np.full_like(s, alpha), s, face=face)
        u, v = orthographic_project(xyz, view_axis=face)
        ax.plot(u, v, linewidth=2.0, color=color)
    for beta in [-np.pi/4, np.pi/4]:
        xyz = gnomonic_equiangular_to_xyz(s, np.full_like(s, beta), face=face)
        u, v = orthographic_project(xyz, view_axis=face)
        ax.plot(u, v, linewidth=2.0, color=color)
