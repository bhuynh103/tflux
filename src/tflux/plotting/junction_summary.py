from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pathlib import Path
import tflux.pipeline.config as config
import tflux.plotting.fft as fft
import tflux.plotting.points as points
import tflux.plotting.grids as grids
import tflux.plotting.linreg as reg
from tflux.dtypes import Junction
from tflux.utils.logging import get_logger

logger = get_logger(__name__)


def plot_junction_summary_3x3(junc: Junction) -> plt.Figure:
    # 3 x 3 summary subplots
    fig, axs = plt.subplots(3, 3, figsize=(11, 11), layout='constrained')
    axs_flat = axs.flatten()

    axs_flat[0] = points.plot_vertices_3d(
        junc.vertices,
        cmap=config.cmap1,
        title='a',
        ax=axs_flat[0],
    )

    axs_flat[1] = grids.plot_xt_surface(
        junc.grid,
        cmap=config.cmap1,
        ax=axs_flat[1],
    )

    # fft.plot_amplitude_distribution(
    #     junc.grid,
    #     bins=50,
    #     cmap=config.cmap1,
    #     ax=axs[0, 2]
    # )

    axs_flat[2] = fft.plot_3d_fft(
        junc.mesh,
        log=True,
        log_residuals=False,
        include_best_fit=True,
        ax=axs_flat[2],
    )

    axs_flat[3], axs_flat[6] = fft.plot_fft_vs_q_omega(
        junc.fft.z_tilde,
        ax1=axs_flat[3],
        ax2=axs_flat[6],
    )

    reg.plot_2d_fft_slope(junc.linreg_w, ax=axs_flat[3])
    reg.plot_2d_fft_slope(junc.linreg_q, ax=axs_flat[6])

    axs_flat[4], axs_flat[7] = fft.plot_fft_vs_q_omega(
        junc.fft.z_tilde,
        ax1=axs_flat[4],
        ax2=axs_flat[7],
        scale='log',
    )

    axs_flat[5] = reg.plot_2d_fft_slope(junc.linreg_w, ax=axs_flat[5], scale='log')
    axs_flat[8] = reg.plot_2d_fft_slope(junc.linreg_q, ax=axs_flat[8], scale='log')

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    for ax, l in zip(axs.flatten(), letters):
        ax.set_title(f'{l}', y=1.05)
    fig.suptitle(f'{junc}')

    # plt.subplots_adjust(right=1.1, wspace=0.7, hspace=0.7)
    # plt.show()

    return fig


def plot_junction_summary_2x2(junc: Junction) -> plt.Figure:
    # 2 x 2 FFT summary subplots
    fig, axs = plt.subplots(2, 2, figsize=(11, 11), squeeze=True) # sharey = 'row'
    axs_flat = axs.flatten()

    fft.plot_fft_vs_q_omega(junc.fft.z_tilde,
                                     ax1=axs_flat[0],
                                     ax2=axs_flat[2])

    fft.plot_fft_vs_q_omega(junc.fft.z_tilde,
                                     ax1=axs_flat[1],
                                     ax2=axs_flat[3],
                                     scale='log')

    reg.plot_2d_fft_slope(junc.linreg_w, ax=axs_flat[1])
    reg.plot_2d_fft_slope(junc.linreg_q, ax=axs_flat[3])

    axs_flat[0] = fft.plot_3d_fft(
        junc.mesh,
        log=True,
        log_residuals=False,
        include_best_fit=True,
        ax=axs_flat[0],
    )

    axs_flat[1] = fft.plot_3d_fft(
        junc.mesh,
        log=True,
        log_residuals=True,
        include_best_fit=True,
        ax=axs_flat[1],
    )

    axs_flat[2] = fft.plot_3d_fft(
        junc.mesh,
        log=False,
        log_residuals=False,
        include_best_fit=False,
        ax=axs_flat[2],
    )

    letters = ['e', 'f', 'h', 'i']
    for ax, l in zip(axs.flatten(), letters):
        ax.set_title(f'{l}')
    fig.suptitle(f'{junc}')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # plt.show()

    return fig


def plot_junction_summary_3x3(junc: Junction, output_dir: Path) -> list[plt.Axes]:  # Overloading the first 3x3 function
    logger.debug(f"Ranges | T: {junc.get_range('t')} s, X: {junc.get_range('x') * 1e6} um")
    axs = []
    axs.append(points.plot_junc_3d(junc))   # C
    axs.append(grids.plot_xt_surface(junc)) # D
    axs.append(grids.plot_xt_surface_projected(junc, over='x')) # E
    axs.append(grids.plot_xt_surface_projected(junc, over='t')) # F
    axs.append(fft.plot_3d_fft(junc.mesh, log=True, log_residuals=False, include_best_fit=True)) # G
    # axs.extend(fft.plot_fft_vs_q_omega(junc.fft.z_tilde))
    # axs.append(reg.plot_2d_fft_slope(junc.linreg_q))
    # axs.append(reg.plot_2d_fft_slope(junc.linreg_w))
    axs.extend(fft.plot_fft_vs_q_omega(junc.fft, scale='log'))  # H, I
    axs.append(reg.plot_2d_fft_slope(junc.linreg_w, scale='log'))       # J
    axs.append(reg.plot_2d_fft_slope(junc.linreg_q, scale='log'))       # K
    # axs.append(grids.plot_qw_surface(junc))

    logger.debug(axs)

    out_subdir = output_dir / f"C{junc.cell_index}" / f"J{junc.roi_index}"
    letters = 'cdefghijkl' # 'abcdefghijkl'
    
    seen_figs = set()
    for ax, letter in zip(axs, letters):
        fig = ax.figure
        if id(fig) not in seen_figs:
            seen_figs.add(id(fig))
            png_name = f'C{junc.cell_index}-J{junc.roi_index}_{letter}_summary.png'
            fig.savefig(out_subdir / png_name, transparent=True)
    plt.close('all')

    return axs