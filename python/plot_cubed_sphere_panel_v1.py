# Gnomonic equiangular cubed-sphere panel grid (single panel, +X face)
# The plot shows coordinate lines crowding/converging toward the panel edges.
import numpy as np
import matplotlib.pyplot as plt

def gnomonic_equiangular_to_xyz(alpha, beta, face="+X"):
    """
    Map equiangular gnomonic coordinates (alpha, beta) in [-pi/4, pi/4] to
    Cartesian coordinates (x, y, z) on the unit sphere for a given cube face.

    Faces:
      "+X", "-X", "+Y", "-Y", "+Z", "-Z"
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

    # Normalize to the unit sphere
    inv_norm = 1.0 / np.sqrt(X*X + Y*Y + Z*Z)
    return X*inv_norm, Y*inv_norm, Z*inv_norm

def orthographic_project(face_xyz, view_axis="+X"):
    """
    Orthographic projection onto the plane normal to view_axis.
    Returns 2D coordinates for plotting.
    """
    x, y, z = face_xyz
    if view_axis == "+X" or view_axis == "-X":
        return y, z
    if view_axis == "+Y" or view_axis == "-Y":
        return x, z
    if view_axis == "+Z" or view_axis == "-Z":
        return x, y
    raise ValueError("Invalid view axis")

def plot_panel_grid(face="+X", n_lines=15, n_pts=400):
    # Uniformly spaced equiangular coordinates (in radians)
    a = np.linspace(-np.pi/4, np.pi/4, n_lines)
    s = np.linspace(-np.pi/4, np.pi/4, n_pts)

    # Prepare plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    # Draw the visible hemisphere outline for context (unit circle under orthographic projection)
    theta = np.linspace(0, 2*np.pi, 720)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1, alpha=0.5)

    # Plot alpha = const (vary beta)
    for alpha in a:
        alpha_arr = np.full_like(s, alpha)
        xyz = gnomonic_equiangular_to_xyz(alpha_arr, s, face=face)
        u, v = orthographic_project(xyz, view_axis=face)  # look along the face normal
        ax.plot(u, v, linewidth=0.8)

    # Plot beta = const (vary alpha)
    for beta in a:
        beta_arr = np.full_like(s, beta)
        xyz = gnomonic_equiangular_to_xyz(s, beta_arr, face=face)
        u, v = orthographic_project(xyz, view_axis=face)
        ax.plot(u, v, linewidth=0.8)

    # Panel boundary (|alpha|=pi/4 and |beta|=pi/4)
    for alpha in [-np.pi/4, np.pi/4]:
        xyz = gnomonic_equiangular_to_xyz(np.full_like(s, alpha), s, face=face)
        u, v = orthographic_project(xyz, view_axis=face)
        ax.plot(u, v, linewidth=1.5)
    for beta in [-np.pi/4, np.pi/4]:
        xyz = gnomonic_equiangular_to_xyz(s, np.full_like(s, beta), face=face)
        u, v = orthographic_project(xyz, view_axis=face)
        ax.plot(u, v, linewidth=1.5)

    ax.set_xlabel("u (orthographic)")
    ax.set_ylabel("v (orthographic)")
    ax.set_title(f"Gnomonic Equiangular Grid on Cubed-Sphere Panel {face}\n(orthographic view along {face})")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    plt.show()

# Generate the plot for a single panel (+X). Increase n_lines to densify the grid.
plot_panel_grid(face="+X", n_lines=17, n_pts=600)
