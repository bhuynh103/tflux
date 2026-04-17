import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from codetiming import Timer

import tflux.io.paths as paths
import tflux.io.obj_reader as obj_reader
from tflux.io.png_to_pdf import pngs_to_pdf
import tflux.pipeline.config as config
import tflux.preprocessing.grid_utils as grid_utils
import tflux.preprocessing.vertices_utils as vertices_utils
import tflux.preprocessing.kmean_norms as kmean_norms
import tflux.analysis.slope_analyzer as slope_analyzer
from tflux.plotting.junction_summary import plot_junction_summary_3x3
from tflux.plotting.points import plot_cell_3d_with_norms
from tflux.plotting.sample_slope_hist import plot_gradient_histograms, plot_all_gradient_histograms, plot_linreg_fits, plot_linreg_hist
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
    if junc.grid is not None and junc.grid.percent_zero > config.PERCENT_ZERO_THRESHOLD:
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
        junc = clean_junction(junc=junc)
        label_junction(junc=junc, cell=cell, sample=sample)

    sample.append_cell(cell=cell)
    sample.append_junctions(juncs=junctions)
    return


def get_files_from_directory(data_dir_path: Path):
    if not data_dir_path.exists() or not data_dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {data_dir_path}")
    
    pkl_files = sorted(data_dir_path.glob("*.pkl"))
    if len(pkl_files) > 1:
        logger.warning(f"Multiple .pkl files found in {data_dir_path}, ignoring and processing OBJ files.")
    elif len(pkl_files) == 1:
        logger.info(f"Found {pkl_files[0]}.")
        if input("Load from pickle instead of processing OBJ files? [y/N]: ").strip().lower() == "y":
            return pkl_files

    obj_files = sorted(data_dir_path.glob("*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {data_dir_path}")
    logger.info(f"Found {len(obj_files)} OBJ files in {data_dir_path}.")
    if input("Proceed with processing OBJ files? [y/N]: ").strip().lower() != "y":
        raise RuntimeError("Aborted by user.")
    return obj_files


def _create_output_dirs(output_dir_path: Path, sample: Sample):
    cell_dir = output_dir_path / "cells"
    for cell_index in range(len(sample.cells)):
        (cell_dir / f"C{cell_index}").mkdir(parents=True, exist_ok=True)
        for junc in sample.cells[cell_index].junctions:
            if junc.roi_index != -1 or config.include_bad_junctions_in_summary:
                (cell_dir / f"C{cell_index}" / f"J{junc.roi_index}").mkdir(parents=True, exist_ok=True)
    return cell_dir


def _load_sample(data_dir_path: Path) -> Sample:
    """Load Sample from .pkl or process .obj files. Saves .pkl if config.save_pickle."""
    files = get_files_from_directory(data_dir_path=data_dir_path)

    if files[0].suffix == ".pkl":
        logger.info(f"Loading sample from pickle: {files[0]}")
        with open(files[0], "rb") as f:
            return pickle.load(f)
    
    sample = process_files(files)

    if config.save_pickle:
        pkl_out = data_dir_path / "sample.pkl"
        with open(pkl_out, "wb") as f:
            pickle.dump(sample, f)
        logger.info(f"Sample saved to {pkl_out}")
    return sample


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
        logger.info(f"Summarizing {len(cell.junctions)} junctions in cell {cell.cell_index}")
        for junc in cell.junctions:
            if junc.roi_index != -1 or config.include_bad_junctions_in_summary:
                # fig = plot_junction_summary_3x3(junc=junc)  # TODO: fix bug and verify fft plots are correct
                # png_name = f'C{cell.cell_index}-J{junc.roi_index}_3x3summary.png'
                # fig.savefig(summary_dir / f"C{cell.cell_index}" / f"J{junc.roi_index}" / png_name)
                # plt.close(fig)
                plot_junction_summary_3x3(junc=junc, output_dir=summary_dir)


### PIPELINE START ###
def run_pipeline(data_dir_path: Path, output_dir_path: Path, sample_label: str = None) -> Sample: # output_dir_path is root of output dir, must enter sample_label subdir

    if not data_dir_path.exists() or not data_dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {data_dir_path}")
    
    if not output_dir_path.exists() or not output_dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {output_dir_path}")

    # Check for .pkl and .obj files, prioritize .pkl if user confirms.
    sample = _load_sample(data_dir_path)

    logger.info("="*60)
    logger.info("Starting tflux pipeline")
    logger.info("="*60)

    output_dir_path = output_dir_path / sample_label
    cell_dir = _create_output_dirs(output_dir_path, sample)

    # Only analyze valid junctions
    metrics_csv_path = slope_analyzer.save_slopes_to_csv(sample, output_dir=output_dir_path)

    if config.save_average_slope_csv:
        slope_analyzer.average_sample_slopes(sample, slopes=None, output_dir=output_dir_path)

    if config.make_junc_summary:
        with Timer(text="Summarized junctions: {:.3f}s", logger=logger.info):
            summarize_sample_junctions(summary_dir=cell_dir, sample=sample)
            # plot_junction_summary_3x3(junc=sample.cells[0].junctions[0], output_dir=cell_dir)  # Debug
            if config.make_pdfs:
                pngs_to_pdf(
                    input_dir=cell_dir, 
                    output_path=Path(cell_dir / f"{output_dir_path.name}-junc_summaries.pdf"),
                    pattern="*/*/*summary.png"
                )

    if config.make_histogram:
        hist_dir = output_dir_path / "histograms"

        # fig = plot_gradient_histograms(csv_path=metrics_csv_path, title=data_dir_path) 
        # png_name = f'{sample_label}_hist.png'
        # fig.savefig(hist_dir / png_name)
        # plt.close(fig)

        fig = plot_linreg_fits(sample)
        png_name = f'{sample_label}_linreg.png'
        fig.savefig(hist_dir / png_name)
        plt.close(fig)

        fig = plot_linreg_hist(sample)
        png_name = f'{sample_label}_hist.png'
        fig.savefig(hist_dir / png_name)
        plt.close(fig)
    
    if config.save_cells:
        with Timer(text="Saved junctions to png and pdf: {:.3f}s", logger=logger.info):
            for cell in sample.cells:
                fig = plot_cell_3d_with_norms(cell=cell, title=cell)    # Fig 1B
                png_name = f'C{cell.cell_index}_render.png'
                fig.savefig(cell_dir / f"C{cell.cell_index}" / png_name, transparent=True)
                plt.close(fig)
            if config.make_pdfs:
                pngs_to_pdf(
                    input_dir=cell_dir, 
                    output_path=Path(cell_dir / f"{output_dir_path.name}-cells.pdf"),
                    pattern="*/*render.png"
                )
            
    return sample
