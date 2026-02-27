import matplotlib.pyplot as plt
from tflux.dtypes import Grid
from tflux.pipeline import config


def plot_xt_surface(grid: Grid, cmap=config.cmap1, ax=None):
    
    surface = grid.z
    x_range = grid.get_grid_range('x')
    t_range = grid.get_grid_range('t')
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    im = ax.imshow(surface.T, cmap=cmap, origin='lower', aspect='auto', extent=[0, x_range, 0, t_range])
    fig.colorbar(im, ax=ax, label='Amplitude (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('T (s)')
    
    return


def plot_amplitude_distribution(grid: Grid, bins=50, cmap=config.cmap1, ax=None):
    surface = grid.z
    amplitudes = surface.flatten()

    if ax is None:
        fig = plt.figure(figsize=(6, 4), dpi=300)
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