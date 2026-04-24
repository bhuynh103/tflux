import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from tflux.dtypes import Grid, GridFFT, Junction
from tflux.pipeline import config


def plot_xt_surface(junc: Junction, ax=None):
    grid: Grid = junc.grid
    surface = grid.z
    surface_scaled = surface * 1e6
    x_range = grid.get_grid_range('x')
    x_range_scaled = x_range * 1e6
    t_range = grid.get_grid_range('t')

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(
        surface_scaled.T,
        origin='lower',
        aspect='auto',
        extent=[0, x_range_scaled, 0, t_range],
        cmap='YlOrRd',
        interpolation='antialiased',
    )
    ax.set_xlabel(u'X (μm)', labelpad=-10)
    ax.set_ylabel('T (s)', labelpad=-20)
    ax.set_xticks([x_range_scaled])
    ax.set_yticks([0, t_range + 1])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(u'Y (μm)', fontsize=32)

    return ax


def plot_xt_surface_projected(junc: Junction, over=None, ax=None):
    # TODO: Consider splitting up this function
    grid: Grid = junc.grid
    surface = grid.z
    cmap_ax1 = mcolors.LinearSegmentedColormap.from_list(
        "red", [(255/255, 80/255, 80/255), (139/255, 0/255, 0/255)]
    )
    cmap_ax2 = mcolors.LinearSegmentedColormap.from_list(
        "orange", [(255/255, 180/255, 50/255), (180/255, 90/255, 0/255)]
    )
        
    shape = surface.shape # (x, t)
    x_length = shape[0]
    t_length = shape[1]

    surface_scaled = surface * 1e6
    dim_range = grid.get_grid_range(over)
    if over == 'x':
        dim_range = dim_range * 1e6

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if over == 'x':
        dim = np.linspace(0, dim_range, x_length)
        indices = range(0, 3*(t_length//5), t_length//5)
        for k, i in enumerate(indices):
            t = k / (len(indices) - 1)
            ax.plot(
                dim,
                surface_scaled[:, i],
                color=cmap_ax1(t),
                label=f'{i}',
            )
        ax.set_xlabel(u'X (μm)', labelpad=0)
        ax.set_ylabel(u'Y (μm)', labelpad=0)
        ax.legend(title='Time (s)', loc='upper left')
    elif over == 't':
        dim = np.linspace(0, dim_range, t_length)
        indices = range(0, 3*(x_length//5), x_length//5)
        for k, i in enumerate(indices):
            t = k / (len(indices) - 1)
            ax.plot(
                dim,
                surface_scaled[i],
                color=cmap_ax2(t),
                label=f'{(i * config.dx * 1e6):.0f}',
            )
        ax.set_xlabel(u'T (s)', labelpad=0)
        ax.set_ylabel(u'Y (μm)', labelpad=0)
        ax.legend(title=u'X (μm)', loc='upper left')

    # ax.set_xticks([x_range_scaled])
    # ax.set_yticks([0, t_range + 1])

    # cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    # cbar.set_label(u'Y (μm)', fontsize=32)

    return ax

