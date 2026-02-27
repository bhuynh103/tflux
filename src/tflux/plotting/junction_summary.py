import matplotlib.pyplot as plt
import tflux.pipeline.config as config
import tflux.plotting.fft as fft
import tflux.plotting.points as points
import tflux.plotting.grids as grids
import tflux.plotting.linreg as reg
from tflux.dtypes import Junction

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