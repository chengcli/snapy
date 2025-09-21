import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

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

def orthographic_project(face_xyz, view_dir=np.array([1,0,0])):
    """Orthographic projection onto the plane normal to view_dir."""
    x, y, z = face_xyz
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

def visible_segments(u, v, depth, vis_mask):
    segs = []
    start = None
    for i in range(len(u)):
        if vis_mask[i] and start is None:
            start = i
        elif (not vis_mask[i]) and (start is not None):
            segs.append((u[start:i], v[start:i], depth[start:i]))
            start = None
    if start is not None:
        segs.append((u[start:], v[start:], depth[start:]))
    return segs

def plot_on_face(ax, alpha, beta, face="+X", view_dir=np.array([1,0,0]),
                 **kwargs):
    x,y,z = gnomonic_equiangular_to_xyz(alpha, beta, face=face)
    # r·V (nearness for ortho)
    depth = x*view_dir[0] + y*view_dir[1] + z*view_dir[2]

    vis = depth > 0 # front hemisphere
    u,v = orthographic_project((x,y,z), view_dir=view_dir)
    segs = visible_segments(u, v, depth, vis)
    segs.sort(key=lambda seg: np.max(seg[2]) if seg[2].size else -1)
    for uu, vv, dd in segs:
        ax.plot(uu, vv, zorder=np.max(dd) if dd.size else 0, **kwargs)

def scatter_on_face(ax, alpha, beta, face="+X", view_dir=np.array([1,0,0]),
                    **kwargs):
    x,y,z = gnomonic_equiangular_to_xyz(alpha, beta, face=face)
    # r·V (nearness for ortho)
    depth = x*view_dir[0] + y*view_dir[1] + z*view_dir[2]

    vis = depth > 0 # front hemisphere
    u,v = orthographic_project((x,y,z), view_dir=view_dir)
    ax.scatter(u[vis], v[vis], zorder=np.max(depth[vis]), **kwargs)

def draw_panel_grid(ax, face="+X", N=8, nghost=3, n_pts=800,
                    view_dir=np.array([1,0,0]),
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


    # alpha = const lines
    for i, alpha in zip(idx, alphas):
        # skip the first and the last lines
        if i == idx[0] or i == idx[-1]: continue
        plot_on_face(ax, np.full_like(s, alpha), s, face=face,
                     view_dir=view_dir,
                     linewidth=linewidth, linestyle=linestyle, color=color)

    # beta = const lines
    for j, beta in zip(idx, alphas):
        # skip the first and the last lines
        if j == idx[0] or j == idx[-1]: continue
        plot_on_face(ax, s, np.full_like(s, beta), face=face,
                     view_dir=view_dir,
                     linewidth=linewidth, linestyle=linestyle, color=color)

    # cell centers (solid dots)
    center_a, center_b = np.meshgrid(centers, centers)
    scatter_on_face(ax, center_a.flatten(), center_b.flatten(),
                    face=face, view_dir=view_dir,
                    s=10, facecolors=facecolor, edgecolors=color)

    # Interior panel boundary (bold): |alpha|=pi/4 and |beta|=pi/4
    for alpha in [-np.pi/4, np.pi/4]:
        plot_on_face(ax, np.full_like(s, alpha), s, face=face,
                     view_dir=view_dir,
                     linestyle="-.", linewidth=1.6, color=color)
    for beta in [-np.pi/4, np.pi/4]:
        plot_on_face(ax, s, np.full_like(s, beta), face=face,
                     view_dir=view_dir,
                     linestyle="-.", linewidth=1.6, color=color)

def draw_single_panel(ax, face="+X", N=8, nghost=3, color='C0',
                      view_dir=np.array([1,0,0])):
    # all grid lines including ghosts
    draw_panel_grid(ax, face=face, N=N, nghost=nghost, color=color,
                    view_dir=view_dir)

    # interior grid lines only
    draw_panel_grid(ax, face=face, N=N, nghost=0, view_dir=view_dir,
                    linestyle='-', linewidth=1.2,
                    facecolor=color, color=color)

def draw_panel_seam(ax, N=8, nghost=3):
    # Use view along bisector of +X and +Y directions, i.e. (1,1,0).
    view_dir = np.array([1.0,1.0,0.0])

    draw_single_panel(ax, "+X", N=N, nghost=nghost,
                      view_dir=view_dir, color='C0')
    draw_single_panel(ax, "+Y", N=N, nghost=nghost,
                      view_dir=view_dir, color='C1')

def draw_panel_corner(ax, N=8, nghost=3):
    # View along the cube-space diagonal to center the +X/+Y/+Z corner
    view_dir = np.array([1.0,1.0,1.0])

    draw_single_panel(ax, "+X", N=N, nghost=nghost,
                      view_dir=view_dir, color='C0')
    draw_single_panel(ax, "+Y", N=N, nghost=nghost,
                      view_dir=view_dir, color='C1')
    draw_single_panel(ax, "+Z", N=N, nghost=nghost,
                      view_dir=view_dir, color='C2')
