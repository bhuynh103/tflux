from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import tflux.io.paths as paths
import tflux.io.obj_reader as obj_reader
import tflux.pipeline.config as config
import tflux.preprocessing.grid_utils as grid_utils
import tflux.preprocessing.vertices_utils as vertices_utils
import tflux.preprocessing.kmean_norms as kmean_norms
import tflux.analysis.slope_analyzer as slope_analyzer
from tflux.plotting.junction_summary import plot_junction_summary_3x3
from tflux.plotting.points import plot_cell_3d
from tflux.plotting.sample_slope_hist import plot_gradient_histograms, plot_all_gradient_histograms
from tflux.dtypes import Sample, Cell, Junction, GridFFT, Grid, Mesh, LinReg
from tflux.utils.logging import get_logger

logger = get_logger(__name__)


# Preprocessing Junction into Grid and Mesh
def grid_junction(junc: Junction) -> Junction:
    logger.info(f"Analyzing junction {junc.roi_index} with {len(junc.vertices)} vertices")

    logger.debug(f"Gridding junction")
    junc.grid = grid_utils.grid_xt(junc)
    logger.debug(f'Original grid size x: {len(junc.grid.x)}, t: {len(junc.grid.t)}')

    logger.debug(f"Interpolating zeros")
    junc.grid = grid_utils.interpolate_zeros(junc.grid)

    logger.debug(f"Trimming grid with crop_percent={config.CROP_PERCENT})")
    junc.grid = grid_utils.trim_grid(junc.grid, crop_percent=config.CROP_PERCENT)
    logger.debug(f'Trimmed grid size x: {len(junc.grid.x)}, t: {len(junc.grid.t)}')
    
    return junc


def analyze_junction_spectrum(junc: Junction) -> Junction:
    junc.fft = grid_utils.fourier_transform(junc.grid, shift_fft=True, square_fft=True)
    logger.debug(f'Trimmed fft grid size q: {len(junc.fft.q)}, w: {len(junc.fft.w)}')
    junc.linreg_q, junc.linreg_w = grid_utils.linreg_on_fft(junc.fft)
    
    logger.debug(f"Constructing mesh from fft")
    junc.mesh = grid_utils.fft_to_mesh(junc.fft)

    logger.debug(f"Applying masks to mesh.")
    junc.mesh = junc.mesh.apply_masks(denoise=True)  # Slice above positive frequency and below noise floor

    logger.debug(f"Finding log-log gradient")
    junc.mesh = junc.mesh.find_loglog_gradient()

    return junc


def clean_junction(junc: Junction) -> Junction:
    junc = vertices_utils.reorient_junction(junc)
    junc = grid_junction(junc)
    junc = analyze_junction_spectrum(junc)
    return junc


def label_junction(junc: Junction, cell: Cell, sample: Sample):
    junc.cell_index = cell.cell_index
    if junc.grid is not None and junc.grid.percent_zero > 0.20:
        logger.info(f"Omitting sparse junction at roi_index {junc.roi_index} with {junc.grid.percent_zero * 100:.2f}% zeroes.")
        junc.roi_index = -1
    else:
        junc.sample_index = len(sample.valid_juncs)
        sample.valid_juncs.append(junc)


def find_junctions_from_file(file_index: int, file_path: Path, sample: Sample):
    # Assumes each file has one cell
    cell = Cell(
        source_file=file_path,
        cell_index=file_index,
    )

    junctions: list[Junction] = kmean_norms.extract_junctions(
        Path(file_path),
        k=config.K_MEANS,
        smooth_iter=config.SMOOTH_ITER,
        lam=config.LAMBDA,
        min_island_faces=config.MIN_ISLAND_FACES,
        normal_weight=config.NORMAL_WEIGHT,
        geom_weight=config.GEOM_WEIGHT,
        cell=cell
    )

    for junc in cell.junctions:
        clean_junction(junc=junc)
        label_junction(junc=junc, cell=cell, sample=sample)

    sample.append_cell(cell=cell)
    sample.append_junctions(juncs=junctions)
    return


def get_files_from_directory(data_dir_path: Path):
    if not data_dir_path.exists() or not data_dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {data_dir_path}")

    obj_files = sorted(data_dir_path.glob("*.obj"))

    if not obj_files:
        logger.warning(f"No OBJ files found in directory: {data_dir_path}")
        return None

    return obj_files


def process_files(files: list[Path]):
    sample = Sample()
    for file_index, file_path in enumerate(files):
        try:
            # Updates Sample with new cell entry, junction entries, and valid_junction entries.
            logger.info(f"\nOpening file {file_index}/{len(files)}")
            find_junctions_from_file(file_index=file_index, file_path=file_path, sample=sample)
        except Exception as e:
            logger.error(f"Failed to process file {file_path.name}: {e}")
            raise
    return sample


def summarize_sample_junctions(summary_dir: Path, sample: Sample):
    for cell in sample.cells:
        for junc in cell.junctions:
            if junc.roi_index != -1 or config.include_bad_junctions_in_summary:
                fig = plot_junction_summary_3x3(junc=junc)  # TODO: fix bug and verify fft plots are correct
                png_name = f'C{cell.cell_index}-J{junc.roi_index}_3x3summary.png'
                fig.savefig(summary_dir / png_name)
                plt.close(fig)


### PIPELINE START ###
def run_pipeline(data_dir_path: Path = None, output_dir_path: Path = None, sample_label: str = None) -> None:
    logger.info("="*60)
    logger.info("Starting tflux pipeline")
    logger.info("="*60)

    files = get_files_from_directory(data_dir_path=data_dir_path)
    sample = process_files(files)

    # Only analyzes valid junctions
    metrics_csv_path = slope_analyzer.save_slopes_to_csv(sample, output_dir=output_dir_path)

    if config.save_average_slope_csv:
        slope_analyzer.average_sample_slopes(sample, slopes=None, output_dir=output_dir_path)

    if config.make_junc_summary:
        junction_summary_dir = output_dir_path / "junction_summaries"
        summarize_sample_junctions(summary_dir=junction_summary_dir, sample=sample)
        pngs_to_pdf(input_dir=junction_summary_dir, output_path=Path(junction_summary_dir / "junc_summaries.pdf"))

    if config.make_histogram:
        hist_dir = output_dir_path / "histograms"
        fig = plot_gradient_histograms(csv_path=metrics_csv_path, title=data_dir_path) 
        png_name = f'{sample_label}_hist.png'
        fig.savefig(hist_dir / png_name)
        plt.close(fig)
    
    if config.save_cells:
        from tflux.io.png_to_pdf import pngs_to_pdf
        for cell in sample.cells:
            cell_dir = output_dir_path / "cells"
            fig = plot_cell_3d(cell=cell, title=cell)
            png_name = f'C{cell.cell_index}.png'
            fig.savefig(cell_dir / png_name)
            plt.close(fig)
        pngs_to_pdf(input_dir=cell_dir, output_path=Path(cell_dir / "cells.pdf"))
            
    return
