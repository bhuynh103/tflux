import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from tflux.plotting.axes import _ensure_ax_3d
from tflux.dtypes import Cell


def plot_vertices_3d(vertices, cmap=None, title=None, ax=None): # Moved to points.py
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


def plot_cell_3d(cell: Cell, title=None, ax=None): # Moved to points.py
    """ Plot the cell in 3D space (t, x, y). """

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
        ax.scatter(t, y, x, c=rgba, s=0.075)

    return ax.figure