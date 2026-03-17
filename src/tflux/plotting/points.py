import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator
from matplotlib.colors import to_rgb
from tflux.plotting.axes import _ensure_ax_3d
from tflux.dtypes import Cell
from tflux.utils.logging import get_logger

logger = get_logger(__name__)

mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.labelsize': 24,
    'axes.titlesize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.dpi': 300,        # publication quality
    'savefig.dpi': 300
})


try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False


def to_numpy(arr):
    """Convert cupy array to numpy if needed."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def get_components(array):
    return array[:, 0], array[:, 1], array[:, 2]


def plot_vertices_3d(vertices, cmap=None, title=None, ax=None):
    """ Plot the vertices in 3D space (t, x, y). """
    t, y, x = vertices[:, 0], vertices[:, 1] * 1e6, vertices[:, 2] * 1e6

    if ax is None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        ax = _ensure_ax_3d(ax, fig)

    # Labels and title
    ax.set_xlabel("T (s)")
    ax.set_ylabel(u"X (μm)")
    ax.set_zlabel(u"Y (μm)")
    ax.set_title(f"{title}")
    ax.set_box_aspect(None, zoom=1)
    
    y_mid = (max(y) + min(y)) / 2
    x_range = max(x) - min(x)
    ax.set_zlim(y_mid - x_range/2, y_mid + x_range/2)
    
    if cmap is None:
        ax.scatter(t, x, y, c='gray', alpha=0.5)
    else:
        ax.scatter(t, x, y, c=y, cmap=cmap, s=5)
    
    return ax


def plot_cell_3d(cell: Cell, title=None, ax=None):
    """ Plot the cell in 3D space (t, x, y). """

    if ax is None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        ax = _ensure_ax_3d(ax, fig)

    # Labels and title
    ax.set_xlabel("T (s)")
    ax.set_ylabel(u"Y (μm)")
    ax.set_zlabel(u"X (μm)")
    ax.set_title(f"{title}")
    ax.set_box_aspect(None, zoom=1)

    # y_mid = (max(y) + min(y)) / 2
    # x_range = max(x) - min(x)
    # ax.set_zlim(y_mid - x_range/2, y_mid + x_range/2)

    color_dict = {
        -1: "#919191",
        0: "#ff0000",
        1: "#0055ff",
        2: "#00ff00",
        3: "#5500ff",
    }

    vertices_list = [junc.original_vertices for junc in cell.junctions]
    roi_indices = [junc.roi_index for junc in cell.junctions]

    # Collect all points to compute global depth range
    all_points = np.vstack([v for v in vertices_list])
    a, b, c = 1.0, 10.0, 10.0
    base_brightness = 0.2
    all_t = all_points[:, 0]
    all_y = all_points[:, 1] * 1e6
    all_x = all_points[:, 2] * 1e6
    depth_all = a * all_t + b * all_y + c * all_x          # proxy for view depth
    d_min, d_max = depth_all.min(), depth_all.max()

    def depth_rgba(vertices, base_color):
        t, y, x = vertices[:, 0], vertices[:, 1] * 1e6, vertices[:, 2] * 1e6
        depth = a * t + b * y + c * x
        # Normalize depth so that far points are dim (0.0), near points bright (1.0)
        brightness = base_brightness + (1 - base_brightness) * (depth - d_min) / (d_max - d_min + 1e-9)
        rgb = np.array(to_rgb(base_color))
        rgba = np.ones((len(t), 4))
        rgba[:, :3] = rgb * brightness[:, None]
        rgba[:, 3] = 1.0   # fixed alpha
        rgba = np.clip(rgba, 0, 1)
        return t, y, x, rgba

    for vertices, roi_index in zip(vertices_list, roi_indices):
        t, y, x, rgba = depth_rgba(vertices, color_dict[roi_index])
        ax.scatter(t, x, y, c=rgba, s=0.075)    # Plot y in the z-axis

    return ax.figure


def rotate_to_minimize_y(
    vertices_list: list[np.ndarray],
    centroids_list: list[np.ndarray],
    norms_list: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Center all junctions around the global y-x mean, then rotate around the
    t-axis using a shared theta that minimizes y-range on a subset of points.
    """
    def rotate(pts: np.ndarray, cos_t: float, sin_t: float) -> np.ndarray:
        out = pts.copy()
        out[:, 1] = pts[:, 1] * cos_t - pts[:, 2] * sin_t
        out[:, 2] = pts[:, 1] * sin_t + pts[:, 2] * cos_t
        return out

    # Ensure numpy
    vertices_list  = [to_numpy(v) for v in vertices_list]
    centroids_list = [to_numpy(c) for c in centroids_list]
    norms_list     = [to_numpy(n) for n in norms_list]

    # Center around global y-x mean (omit t)
    all_verts = np.vstack(vertices_list)
    global_mean = np.array([0.0, all_verts[:, 1].mean(), all_verts[:, 2].mean()])
    centered_vertices  = [v - global_mean for v in vertices_list]
    centered_centroids = [c - global_mean for c in centroids_list]

    # Find theta that minimizes y-range on a subset
    subset = np.vstack(centered_vertices)[::200]
    best_theta, best_y = 0.0, np.inf
    for deg in range(-90, 90, 10):
        theta = np.radians(deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotated_y = subset[:, 1] * cos_t - subset[:, 2] * sin_t
        if (ptp := np.ptp(rotated_y)) < best_y:
            best_y = ptp
            best_theta = theta

    cos_t, sin_t = np.cos(best_theta), np.sin(best_theta)
    rot_vertices  = [rotate(v, cos_t, sin_t) for v in centered_vertices]
    rot_centroids = [rotate(c, cos_t, sin_t) for c in centered_centroids]
    rot_norms     = [rotate(n, cos_t, sin_t) for n in norms_list]

    return rot_vertices, rot_centroids, rot_norms


# TODO: Continue work on Fig. 1, migrate 2d plots to seaborn
def plot_cell_3d_with_norms(cell: Cell, title=None, ax=None):
    """Plot the cell in 3D space (t, x, y) with face normals."""
    SCALE       = 1e6
    PT_SIZE     = 90
    # colorblind-safe palette for junctions
    palette = sns.color_palette("colorblind", n_colors=4)
    norm_palette = [sns.desaturate(c, 0.7) for c in palette]  # softer for norms

    COLOR      = {-1: "#919191", **{i: mcolors.to_hex(palette[i])      for i in range(4)}}
    NORM_COLOR = {-1: "#919191", **{i: mcolors.to_hex(norm_palette[i]) for i in range(4)}}
    FIG_COLOR   = "#FFFFFF"
    AX_COLOR    = "#FFFFFF"
    PANE_COLOR  = "#F3F3F3"

    def scale_xy(arr):
        out = arr.copy()
        out[:, 1:] *= SCALE
        return out

    # Prepare data
    vertices_list  = [scale_xy(to_numpy(j.original_vertices)) for j in cell.junctions]
    centroids_list = [scale_xy(to_numpy(j.face_centroids))    for j in cell.junctions]
    norms_list     = [scale_xy(to_numpy(j.face_normals))      for j in cell.junctions]
    roi_indices    = [j.roi_index for j in cell.junctions]

    vertices_rot, centroids_rot, norms_rot = rotate_to_minimize_y(
        vertices_list, centroids_list, norms_list
    )

    # Axis limits
    all_pts = np.vstack(vertices_rot)
    all_t, all_y, all_x = all_pts[:, 0], all_pts[:, 1], all_pts[:, 2]
    t_range = ((np.ptp(all_t) + 10) // 10) * 10
    y_range = ((np.ptp(all_y) + 10) // 10) * 10
    x_range = ((np.ptp(all_x) + 10) // 10) * 10
    graph_range = np.sqrt(y_range ** 2 + x_range ** 2)

    mid_y, mid_x = (all_y.max() + all_y.min()) / 2, (all_x.max() + all_x.min()) / 2
    half = max(y_range, x_range) * 1.1 / 2

    # Plotting
    if ax is None:
        fig = plt.figure(figsize=(12, 8), facecolor=FIG_COLOR)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        ax = _ensure_ax_3d(ax, fig)

    point_size = PT_SIZE / len(all_pts)
    for verts, centroids, norms, roi in zip(vertices_rot, centroids_rot, norms_rot, roi_indices):
        t, y, x = verts[:, 0], verts[:, 1], verts[:, 2]
        ax.scatter(t, x, y, c=COLOR[roi], s=point_size)

        # Single quiver arrow at midpoint
        mid = len(centroids) // 2
        ct, cy, cx = centroids[mid, 0], centroids[mid, 1], centroids[mid, 2]
        nt, ny, nx = norms[mid, 0],     norms[mid, 1],     norms[mid, 2]
        ax.quiver(ct, cx, cy, nt, nx, ny,
                  length=graph_range * 0.25, normalize=True,
                  color=NORM_COLOR[roi], linewidth=2.0,
                  alpha=1.0, arrow_length_ratio=0.3)

    ax.view_init(elev=20, azim=-45, roll=0)
    ax.set_box_aspect(None, zoom=0.85)
    ax.set_facecolor(AX_COLOR)
    ax.set_title(title,      fontsize=18, pad=20)
    ax.set_xlabel("T (s)",   fontsize=24)
    ax.set_ylabel(u"X (μm)", fontsize=24)
    ax.set_zlabel(u"Y (μm)", fontsize=24, labelpad=15)
    ax.set_zlim(mid_y - half, mid_y + half)
    ax.set_ylim(mid_x - half, mid_x + half)
    ax.xaxis.set_major_locator(FixedLocator([0, int(t_range)]))
    ax.yaxis.set_major_locator(FixedLocator([int(-x_range/2), int(x_range/2)]))
    ax.zaxis.set_major_locator(FixedLocator([int(-y_range/2), int(y_range/2)]))
    ax.grid(True)

    ax.xaxis.set_pane_color(PANE_COLOR)
    ax.yaxis.set_pane_color(PANE_COLOR)
    ax.zaxis.set_pane_color(PANE_COLOR)

    # Cursed Axes3D tick_params workaround, changes defaults from dict in axis3d.py
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['tick']['outward_factor'] = 0.3  # default 0.1
        axis._axinfo['tick']['inward_factor']  = 0.3  # default 0.2
        axis._axinfo['tick']['linewidth'] = {
            True:  2.0,  # major ticks
            False: 1.0,  # minor ticks
        }

    ax.tick_params(labelsize=20) # Axes3D ignores length and width kwargs

    return fig