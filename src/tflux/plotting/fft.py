import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from matplotlib.colorbar import Colorbar
from tflux.pipeline import config
from tflux.plotting.axes import _ensure_ax_3d
from tflux.dtypes import GridFFT, Mesh
from tflux.utils.logging import get_logger
from tflux.plotting.rcparams import apply_3d_style
from tflux.plotting.plotting_utils import set_3d_axis_ticksize

logger = get_logger(__name__)

def plot_3d_fft(mesh: Mesh, log=False, log_residuals=False, include_best_fit=True, ax=None):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        ax = _ensure_ax_3d(ax, fig)
    
    if not log:
        x = mesh.q
        y = mesh.w
        z = mesh.z
    else:
        x = mesh.log_transform().q
        y = mesh.log_transform().w
        z = mesh.log_transform().z
        if log_residuals:
            z = mesh.log_transform().get_residuals()
            include_best_fit = False
            print("Ignoring best-fit plane for residual plot.")
    
    ax.plot_trisurf(
        x,
        y,
        z,
        cmap='YlOrBr', edgecolor='none', alpha=0.95
    )
        
    # Optionally plot the fitted plane
    if include_best_fit:
        # Create a coarse grid in (x, y) over the data range
        x_min, x_max = x.min() - 0.25, x.max()
        y_min, y_max = y.min() - 0.1, y.max()
        Xp, Yp = np.meshgrid(
            np.linspace(x_min, x_max, 30),
            np.linspace(y_min, y_max, 30),
        )
        Zp = mesh.a * Xp + mesh.b * Yp + mesh.c

        ax.plot_surface(
            Xp, Yp, Zp,
            alpha=0.3,
            edgecolor='none'
        )
    
    # Labels and title
    ax.tick_params(axis='x', pad=15)
    ax.tick_params(axis='y', pad=7.5)
    ax.tick_params(axis='z', pad=15)
    ax.set_xlabel(r"$q \; (m^{-1}$)", labelpad=15)
    ax.set_ylabel(r"$\omega \; (s^{-1}$)", labelpad=15)
    ax.set_zlabel(r"$\langle |\tilde{u}^2| \rangle \; (m^4 s^2)$", labelpad=35)
    ax.set_box_aspect(None, zoom=0.85)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

    # tick_locations = [1, 2, 3] # Corresponding to 10^1, 10^2, 10^3 in log10 space
    # ax.set_zticks(tick_locations)
    formatter = ticker.FuncFormatter(lambda x, pos: f'$10^{{{int(x)}}}$')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(ticker.FixedLocator([4, 6]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-3, -1]))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[2, 4, 5], integer=True))
    
    # mappable = ax.collections[0]
    # cb = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
    # cb.set_label(r"$\langle |\tilde{u}^2| \rangle \; (m^4 s^2)$")
    # cb.locator = MinMaxLocator(z.min(), z.max())
    # cb.formatter = MinMaxFormatter()
    # cb.update_ticks()

    apply_3d_style(ax)
    ax = set_3d_axis_ticksize(ax=ax)
    return ax


def _add_fft_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    cmap: mcolors.LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    label: str,
) -> Colorbar:
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(label, labelpad=8)
    cb.locator = ticker.NullLocator()
    cb.formatter = ticker.NullFormatter()
    cb.update_ticks()
    return cb


def _plot_fft_lines(
    ax: plt.Axes,
    x: np.ndarray,
    lines: np.ndarray,
    cmap: mcolors.LinearSegmentedColormap,
    xlabel: str,
    scale: str | None,
) -> None:
    n_lines = lines.shape[0]
    for i in range(n_lines):
        t = i / (n_lines - 1)
        ax.plot(x, lines[i], c=cmap(t), alpha=1, linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\langle |\tilde{u}^2| \rangle \; (m^4 s^2)$")
    ax.set_facecolor('lightgray')
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')


def plot_fft_vs_q_omega(
    fft: GridFFT,
    ax1: plt.Axes | None = None,
    ax2: plt.Axes | None = None,
    scale: str | None = None,
) -> list[plt.Axes]:
    if not fft.squared or not fft.shifted:
        raise ValueError("FFT must be shifted and squared before plotting.")

    q_positive_mask = fft.q > 0
    omega_positive_mask = fft.w > 0
    log_q = np.log10(fft.q[q_positive_mask])
    log_omega = np.log10(fft.w[omega_positive_mask])

    valid_q_mask = log_q < config.TANGENT_CUTOFF
    valid_omega_mask = log_omega < config.TANGENT_CUTOFF_TIME

    fft_positive = fft.z_tilde[q_positive_mask][:, omega_positive_mask]
    max_m = int(np.sum(valid_q_mask))
    max_n = int(np.sum(valid_omega_mask))

    cmap_ax1 = mcolors.LinearSegmentedColormap.from_list(
        "blue_purple", [(44/255, 14/255, 227/255), (156/255, 14/255, 227/255)]
    )
    cmap_ax2 = mcolors.LinearSegmentedColormap.from_list(
        "green_blue", [(44/255, 168/255, 60/255), (44/255, 168/255, 200/255)]
    )

    fig1, ax1 = (plt.subplots(layout='tight') if ax1 is None else (ax1.get_figure(), ax1))
    fig2, ax2 = (plt.subplots(layout='tight') if ax2 is None else (ax2.get_figure(), ax2))
    # fig1.subplots_adjust(top=0.85) # Must be tight_layout
    # fig2.subplots_adjust(top=0.85)

    _plot_fft_lines(ax1, fft.w[omega_positive_mask], fft_positive[:max_m], cmap_ax1, r"$\omega \; (s^{-1})$", scale)
    _plot_fft_lines(ax2, fft.q[q_positive_mask], fft_positive[:, :max_n].T, cmap_ax2, r"$q \; (m^{-1})$", scale)

    _add_fft_colorbar(fig1, ax1, cmap_ax1, fft.q[q_positive_mask][valid_q_mask].min(), fft.q[q_positive_mask][valid_q_mask].max(), r"$q \; (m^{-1})$")
    _add_fft_colorbar(fig2, ax2, cmap_ax2, fft.w[omega_positive_mask][valid_omega_mask].min(), fft.w[omega_positive_mask][valid_omega_mask].max(), r"$\omega \; (s^{-1})$")

    if scale == 'log':
        ax2.set_xlim(fft.q[q_positive_mask].min() * 0.8, fft.q[q_positive_mask].max() * 1.2)

    return [ax1, ax2]