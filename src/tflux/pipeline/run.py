from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tflux.io.paths as paths
import tflux.pipeline.config as config
import tflux.preprocessing.grid_utils as grid_utils
import tflux.preprocessing.vertices_utils as vertices_utils
import tflux.preprocessing.kmean_norms as kmean_norms
import tflux.io.obj_reader as obj_reader
import tflux.analysis.slope_analyzer as slope_analyzer
from tflux.plotting.junction_summary import plot_junction_summary_3x3
from tflux.plotting.sample_slope_hist import plot_gradient_histograms, plot_all_gradient_histograms
from tflux.dtypes import Sample, Cell, Junction, GridFFT, Grid, Mesh, LinReg
from tflux.utils.logging import get_logger

logger = get_logger(__name__)

# Preprocessing .obj into top and bottom Junctions
def prepare_obj(file: Path) -> tuple[Junction, Junction]:

    vertices: np.ndarray = obj_reader.load_obj(file, element='vertices', relabel=True)
    normals: np.ndarray = obj_reader.load_obj(file, element='normals', relabel=True)
        
    best_vertices = vertices_utils.find_best_orientation(vertices, normals)
    best_vertices[:, 0] *= config.dt # pixels to seconds
    best_vertices[:, 1:] *= config.dx # pixels to meters

    top_half, bottom_half = vertices_utils.slice_vertices(best_vertices)
    
    top_junc = Junction(vertices=top_half, is_top=True)
    bot_junc = Junction(vertices=bottom_half, is_top=False)
    
    return top_junc, bot_junc


# Converting Grid to Mesh, grid shape (288, 598)
def fft_to_mesh(grid_fft: GridFFT) -> Mesh:        
    z2_mesh = grid_fft.z_tilde
    q_mesh, w_mesh = np.meshgrid(grid_fft.q, grid_fft.w, indexing='ij')
    mesh = Mesh(q_mesh, w_mesh, z2_mesh, log_scale=False)
    return mesh


# Converting Grid to LinReg
def linreg_on_fft(grid_fft: GridFFT) -> tuple[LinReg, LinReg]:
    linreg_q = grid_fft.fft_to_linreg_over('q')
    linreg_w = grid_fft.fft_to_linreg_over('w')
    return linreg_q, linreg_w


# Preprocessing Junction into Grid and Mesh
def process_surface(junc: Junction) -> Junction:
    logger.info(f"Gridding junction {junc.roi_index} with {len(junc.vertices)} vertices")
    junc.grid = grid_utils.grid_xt(junc)  # Constructs the Grid object
    logger.debug(f'Original grid size x: {len(junc.grid.x)}, t: {len(junc.grid.t)}')

    logger.info(f"Interpolating zeros")
    junc.grid = grid_utils.interpolate_zeros(junc.grid)

    logger.info(f"Trimming grid with crop_percent={config.CROP_PERCENT})")
    junc.grid = grid_utils.trim_grid(junc.grid, crop_percent=config.CROP_PERCENT)
    logger.debug(f'Trimmed grid size x: {len(junc.grid.x)}, t: {len(junc.grid.t)}')

    junc.fft = junc.grid.fourier_transform(shift_fft=True, square_fft=True)
    logger.debug(f'Trimmed fft grid size q: {len(junc.fft.q)}, w: {len(junc.fft.w)}')
    junc.linreg_q, junc.linreg_w = linreg_on_fft(junc.fft)
    
    logger.info(f"Constructing mesh from fft")
    junc.mesh = fft_to_mesh(junc.fft)  # Contruct the Mesh object

    logger.info(f"Applying masks to mesh.")
    junc.mesh = junc.mesh.apply_masks(denoise=True)  # Slice above positive frequency and below noise floor

    logger.info(f"Finding log-log gradient")
    junc.mesh = junc.mesh.find_loglog_gradient()
    
    return junc


### Batch Processing via Directory ###
def process_files(data_dir_path=None):

    data_dir_path = Path(data_dir_path)

    # Find .obj files
    obj_files = sorted(data_dir_path.glob("*.obj"))
    if not obj_files:
        logger.warning(f"No OBJ files found in directory: {data_dir_path}")
        return None
    
    sample = Sample()
    for file_path in obj_files:
        junctions = kmean_norms.extract_junctions_partitioned(
            Path(file_path),
            k=3,
            lam=0.9,
            min_island_faces=500,
        )

        junctions = [process_surface(junc) for junc in junctions]

        for junc in junctions:
            junc.source_file = file_path

        cell = Cell(junctions=junctions)
        sample.append_junctions(juncs=junctions)
        sample.append_cell(cell=cell)

    return sample


### PIPELINE START ###
def run_pipeline(data_dir_path: Path = None, output_dir_path: Path = None, sample_label: str = None) -> None:
    logger.info("="*60)
    logger.info("Starting tflux pipeline")
    logger.info("="*60)
    # Process files and extract slopes
    sample = process_files(data_dir_path)

    metrics_csv_path = slope_analyzer.save_slopes_to_csv(sample, output_dir=output_dir_path)    # Data saved to slopes.csv in output_dir_path

    if config.save_average_slope_csv:
        slope_analyzer.average_sample_slopes(sample, slopes=None, output_dir=output_dir_path)

    if config.make_junc_summary:
        
        junction_summary_dir = output_dir_path / "junction_summaries"

        for junc in sample.juncs:
            fig = plot_junction_summary_3x3(junc=junc)  # TODO: fix bug and verify fft plots are correct
            png_name = f'{junc.source_file.stem}_J{junc.roi_index}_3x3summary.png'
            fig.savefig(junction_summary_dir / png_name)
            plt.close(fig)
    
    if config.make_histogram:
        hist_dir = output_dir_path / "histograms"
        fig = plot_gradient_histograms(csv_path=metrics_csv_path, title=data_dir_path) 
        png_name = f'{sample_label}_hist.png'
        fig.savefig(hist_dir / png_name)
        plt.close(fig)
    
    return