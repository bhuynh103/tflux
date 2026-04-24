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


def plot_junction_summary(junc: Junction, output_dir: Path) -> list[plt.Axes]:  
    logger.debug(f"Ranges | T: {junc.get_range('t')} s, X: {junc.get_range('x') * 1e6} um")
    axs = []
    axs.append(points.plot_junc_3d(junc))   # C
    axs.append(grids.plot_xt_surface(junc)) # D
    axs.append(grids.plot_xt_surface_projected(junc, over='x')) # E
    axs.append(grids.plot_xt_surface_projected(junc, over='t')) # F
    axs.append(fft.plot_3d_fft(junc.mesh, log=True, log_residuals=False, include_best_fit=True)) # G
    axs.extend(fft.plot_fft_vs_q_omega(junc.fft, scale='log'))  # H, I
    axs.append(reg.plot_2d_fft_slope(junc.linreg_w, scale='log'))       # J
    axs.append(reg.plot_2d_fft_slope(junc.linreg_q, scale='log'))       # K

    logger.debug(axs)

    out_subdir = output_dir / f"C{junc.cell_index}" / f"J{junc.roi_index}"
    letters = 'cdefghijk' # 'cdefghijk'
    
    seen_figs = set()
    for ax, letter in zip(axs, letters):
        fig = ax.figure
        if id(fig) not in seen_figs:
            seen_figs.add(id(fig))
            png_name = f'C{junc.cell_index}-J{junc.roi_index}_{letter}_summary.png'
            fig.savefig(out_subdir / png_name, transparent=True)
    plt.close('all')

    return axs

