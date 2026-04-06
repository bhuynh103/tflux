import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

    grid: Grid = junc.grid
    surface = grid.z
        
    shape = surface.shape # (x, t)
    if over == 'x':
        length = shape[0]
    elif over == 't':
        length = shape[1]
    
    surface_scaled = surface * 1e6
    dim_range = grid.get_grid_range(over)
    if over == 'x':
        dim_range = dim_range * 1e6

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    dim = np.linspace(0, dim_range, length)
    if over == 'x':
        for i in range(0, 200, 50):
            im = ax.plot(
                dim,
                surface_scaled[:, i],
                label=f'{i}',
            )
        ax.set_xlabel(u'X (μm)', labelpad=0)
        ax.set_ylabel(u'Y (μm)', labelpad=0)
        ax.legend(title='Time (s)', loc='upper left')
    elif over == 't':
        for i in range(0, 100, 25):
            im = ax.plot(
                dim,
                surface_scaled[i],
                label=f'{(i * config.dx * 1e6):.0f}'
            )
        ax.set_xlabel(u'T (s)', labelpad=0)
        ax.set_ylabel(u'Y (μm)', labelpad=0)
        ax.legend(title=u'X (μm)', loc='upper left')
    ax.xaxis.set_major_locator(ticker.FixedLocator([0, dim_range]))

    # ax.set_xticks([x_range_scaled])
    # ax.set_yticks([0, t_range + 1])

    # cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    # cbar.set_label(u'Y (μm)', fontsize=32)

    return ax


def plot_qw_surface(junc: Junction, ax=None):

    grid: GridFFT = junc.fft
    if grid.squared:
        surface = grid.z_tilde
    else:
        surface = np.abs(grid.z_tilde) ** 2

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(
        surface.T,
        origin='lower',
        aspect='auto',
        extent=[grid.q.min(), grid.q.max(), grid.w.min(), grid.w.max()],
        cmap=config.cmap1,
        interpolation='antialiased',
    )
    ax.set_xlabel(u'q (m^{-1})', labelpad=-10)
    ax.set_ylabel(u'Hz (s^{-1})', labelpad=-20)
    ax.set_xticks([grid.q.min(), grid.q.max()])
    ax.set_yticks([grid.w.min(), grid.w.max()])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(u'Amplitude Squared * Hz (m^{-2} s^{-1})', fontsize=32)

    return ax


def plot_amplitude_distribution(grid: Grid, bins=50, cmap=config.cmap1, ax=None):
    surface = grid.z
    amplitudes = surface.flatten()

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    n, bins, patches = ax.hist(
        amplitudes,
        bins=bins,
        edgecolor="black",
        linewidth=0.6,
    )
    
    bin_centers = 0.5 * (bins[:-1] - bins[1:])
    col = bins - min(bin_centers)
    col /= max(col)
    
    cm = plt.get_cmap(cmap)
    
    for c, p in zip(col, patches):
        p.set_facecolor(cm(c))
    
    ax.set_xlabel("Amplitude", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)

    return ax