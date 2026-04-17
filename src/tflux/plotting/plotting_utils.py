import numpy as np
from matplotlib.ticker import FixedLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns


palette = sns.color_palette("bright", n_colors=4)
COLOR      = {-1: "#919191", **{i: mcolors.to_hex(palette[i]) for i in range(4)}}
COLOR      = {-1: "#919191", 0: "#FF0000", 1: "#009000", 2: "#3939FF", 3: "#FF7F00"}   
PT_SIZE    = 90
ZOOM       = 1.25


class MinMaxFormatter(ticker.Formatter):
    def __call__(self, x, pos=None):
        if x == 0:
            return "$0$"
        exp = int(np.round(np.log10(abs(x))))
        return f"$10^{{{exp}}}$"


class MinMaxLocator(ticker.Locator):
    def __init__(self, vmin: float, vmax: float):
        self._vmin = min(vmin, vmax)
        self._vmax = max(vmin, vmax)

    def __call__(self):
        return self.tick_values(self._vmin, self._vmax)

    def tick_values(self, vmin, vmax):
        vmax_rounded = 10 ** np.floor(np.log10(abs(self._vmax)))
        vmin_rounded = vmax_rounded / 10
        if vmin_rounded >= vmax_rounded:
            vmin_rounded = vmax_rounded / 10
        return np.array(sorted([vmin_rounded, vmax_rounded]))


def scale_xy(arr):
    SCALE = 1e6
    out = arr.copy()
    out[:, 1:] *= SCALE
    return out


def set_3d_axis_ticks(ax: Axes3D, all_pts: np.ndarray) -> None:
    """
    Compute symmetric FixedLocator ticks from point cloud and apply to all axes.
    Expects all_pts columns as (t, y, x).
    """
    def rounded_range(arr):
        return ((np.ptp(arr) + 10) // 10) * 10

    t_range = rounded_range(all_pts[:, 0])
    y_range = rounded_range(all_pts[:, 1])
    x_range = rounded_range(all_pts[:, 2])
    max_range = max(y_range, x_range)
    half = max_range * 1.1 / 2      # Add 10% padding to ensure all points fit within the limits

    mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2
    mid_x = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2

    ax.xaxis.set_major_locator(FixedLocator([0, int(t_range)]))
    ax.yaxis.set_major_locator(FixedLocator([int(-x_range / 2), int(x_range / 2)]))
    ax.zaxis.set_major_locator(FixedLocator([int(-y_range / 2), int(y_range / 2)]))

    ax.set_ylim(mid_x - half, mid_x + half)
    ax.set_zlim(mid_y - half, mid_y + half)



def letter_annotation(ax, xoffset, yoffset, letter):
    """Add a bold letter annotation in axes-relative coordinates, Axes3D-compatible."""
    try:
        # get axes position in figure coordinates
        pos = ax.get_position()
        fig_x = pos.x0 + xoffset * pos.width
        fig_y = pos.y0 + yoffset * pos.height
        ax.figure.text(fig_x, fig_y, letter, size=12, weight='bold',
                       transform=ax.figure.transFigure)
    except AttributeError:
        ax.text(xoffset, yoffset, letter, transform=ax.transAxes, size=12, weight='bold')


def set_3d_axis_ticksize(ax: Axes3D):
    # Cursed Axes3D tick_params workaround, changes defaults from dict in axis3d.py
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['tick']['outward_factor'] = 0.3  # default 0.1
        axis._axinfo['tick']['inward_factor']  = 0.3  # default 0.2
        axis._axinfo['tick']['linewidth'] = {
            True:  2.0,  # major ticks
            False: 1.0,  # minor ticks
        }
    return ax