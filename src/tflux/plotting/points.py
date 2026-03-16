import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator
from matplotlib.colors import to_rgb
from tflux.plotting.axes import _ensure_ax_3d
from tflux.dtypes import Cell
from tflux.utils.logging import get_logger

logger = get_logger(__name__)


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


def plot_cell_3d_with_norms(cell: Cell, title=None, ax=None):
    """ Plot the cell in 3D space (t, x, y). """
    "TODO: change font size, choose seaborn colors, change ax grid color"

    if ax is None:
        fig = plt.figure(figsize=(10,8), facecolor="#5a6c8d", edgecolor="#FF00C3")
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        ax = _ensure_ax_3d(ax, fig)

    # Labels and title
    ax.set_xlabel("T (s)")
    ax.set_ylabel(u"X (μm)")
    ax.set_zlabel(u"Y (μm)")
    ax.set_title(f"{title}")
    ax.set_facecolor("#5d7fc0")
    fig.set_edgecolor("#FF00C3")
    ax.set_box_aspect(None, zoom=0.85)

    xy_scale_factor = 1e6
    logger.debug(f"Scaling x and y by {xy_scale_factor}")
    xy_step = 1

    color_dict = {
        -1: "#919191",
        0: "#b80000",
        1: "#0055ff",
        2: "#00ff00",
        3: "#5500ff",
    }

    norm_color_dict = {
        -1: "#919191",
        0: "#ff4141",
        1: "#3579ff",
        2: "#48ff48",
        3: "#4b00e2",
    }


    def rotate_to_minimize_y(
        vertices_list: list[np.ndarray],
        centroids_list: list[np.ndarray],
        norms_list: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Center all junctions around the global mean, then rotate around the t-axis
        using a single shared theta to minimize mean y.
        """

        # Ensure all inputs are numpy
        vertices_list  = [to_numpy(v) for v in vertices_list]
        centroids_list = [to_numpy(c) for c in centroids_list]
        norms_list     = [to_numpy(n) for n in norms_list]

        def rotate(pts: np.ndarray, cos_t: float, sin_t: float) -> np.ndarray:
            out = pts.copy()
            out[:, 1] = pts[:, 1] * cos_t - pts[:, 2] * sin_t
            out[:, 2] = pts[:, 1] * sin_t + pts[:, 2] * cos_t
            return out

        all_verts = np.vstack(vertices_list)
        mean_y = all_verts[:, 1].mean()
        mean_x = all_verts[:, 2].mean()
        global_mean = np.array([0.0, mean_y, mean_x])

        # Center
        centered_vertices  = [v - global_mean for v in vertices_list]
        centered_centroids = [c - global_mean for c in centroids_list]

        # Subsample for theta search
        all_centered = np.vstack(centered_vertices)
        subset = all_centered[::200]  # thin subset for speed

        # Search over 30-degree intervals
        best_theta = 0.0
        best_y = np.inf
        for deg in range(-90, 90, 10):
            theta = np.radians(deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotated_y = subset[:, 1] * cos_t - subset[:, 2] * sin_t
            curr_y = np.ptp(rotated_y)
            if curr_y < best_y:
                best_y = curr_y
                best_theta = theta

        cos_t, sin_t = np.cos(best_theta), np.sin(best_theta)

        rot_vertices  = [rotate(v, cos_t, sin_t) for v in centered_vertices]
        rot_centroids = [rotate(c, cos_t, sin_t) for c in centered_centroids]
        rot_norms     = [rotate(n, cos_t, sin_t) for n in norms_list]

        return rot_vertices, rot_centroids, rot_norms


    def scale_xy(coords, scale_factor):
        scaled_coords = coords.copy()
        scaled_coords[:, 1] *= scale_factor # Scale y by 1e6 for better visualization
        scaled_coords[:, 2] *= scale_factor # Scale x by 1e6 for better visualization
        return scaled_coords
    
    # Partition ROIs in list
    vertices_list = [scale_xy(junc.original_vertices, xy_scale_factor) for junc in cell.junctions]
    centroids_list = [scale_xy(junc.face_centroids, xy_scale_factor) for junc in cell.junctions]
    norms_list = [scale_xy(junc.face_normals, xy_scale_factor) for junc in cell.junctions]
    roi_indices = [junc.roi_index for junc in cell.junctions]

    vertices_rotated, centroids_rotated, norms_rotated = rotate_to_minimize_y(vertices_list, centroids_list, norms_list)

    # Collect all points to compute global depth range
    all_points = np.vstack([v for v in vertices_rotated])
    a, b, c = 1.0, 10.0, 10.0
    base_brightness = 0.2
    all_t = all_points[:, 0]
    all_y = all_points[:, 1]
    all_x = all_points[:, 2]
    depth_all = a * all_t + b * all_y + c * all_x          # proxy for view depth
    d_min, d_max = depth_all.min(), depth_all.max()

    def depth_rgba(vertices, base_color):
        t, y, x = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        depth = a * t + b * y + c * x
        # Normalize depth so that far points are dim (0.0), near points bright (1.0)
        brightness = base_brightness + (1 - base_brightness) * (depth - d_min) / (d_max - d_min + 1e-9)
        rgb = np.array(to_rgb(base_color))
        rgba = np.ones((len(t), 4))
        rgba[:, :3] = rgb * brightness[:, None]
        rgba[:, 3] = 1.0   # fixed alpha
        rgba = np.clip(rgba, 0, 1)
        return t, y, x, rgba

    max_range = max(np.ptp(all_y), np.ptp(all_x)) * 1.1
    mid_y = (max(all_y) + min(all_y)) / 2
    mid_x = (max(all_x) + min(all_x)) / 2
    ax.set_zlim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_ylim(mid_x - max_range/2, mid_x + max_range/2)

    t_range = np.ptp(all_t)
    x_range = np.ptp(all_x)
    y_range = np.ptp(all_y) 
    logger.debug(f"t_range: {t_range:.2f}, x_range: {x_range:.2f}, y_range: {y_range:.2f}")
    ax.xaxis.set_major_locator(FixedLocator(locs=list(map(int, [0, t_range]))))
    ax.yaxis.set_major_locator(FixedLocator(locs=list(map(int, [-x_range/2, x_range/2]))))
    ax.zaxis.set_major_locator(FixedLocator(locs=list(map(int, [-y_range/2, y_range/2]))))


    graph_range = np.sqrt(y_range ** 2 + x_range ** 2)
    
    for vertices, centroids, norms, roi_index in zip(vertices_rotated, centroids_rotated, norms_rotated, roi_indices):
        num_arrows = 1
        length = vertices.shape[0]
        start = int(length * 0.5)
        stop = int(length * 0.51)
        step = max(int((stop - start) / num_arrows), 1)

        t, y, x, rgba = depth_rgba(vertices, color_dict[roi_index])

        ct, cy, cx = get_components(centroids)
        nt, ny, nx = get_components(norms)
        # ax.scatter(t[::xy_step], x[::xy_step], y[::xy_step], c=rgba[::xy_step], s=0.0025)
        ax.scatter(t[::xy_step], x[::xy_step], y[::xy_step], c=color_dict[roi_index], s=0.0015)
        ax.quiver(
            to_numpy(ct)[start:stop:step], to_numpy(cx)[start:stop:step], to_numpy(cy)[start:stop:step],
            to_numpy(nt)[start:stop:step], to_numpy(nx)[start:stop:step], to_numpy(ny)[start:stop:step],
            length=graph_range * 0.25,
            alpha=1.0,
            color=norm_color_dict[roi_index],
            linewidth=2.0,
            normalize=True,
            arrow_length_ratio=0.3
        )
    return ax.figure