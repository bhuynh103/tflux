from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tflux.io.paths as paths
import tflux.pipeline.config as config
import tflux.preprocessing.grid_utils as grid_utils
import tflux.preprocessing.vertices_utils as vertices_utils
import tflux.io.obj_reader as obj_reader
import tflux.analysis.slope_analyzer as slope_analyzer
from tflux.plotting.junction_summary import plot_junction_summary_3x3
from tflux.plotting.sample_slope_hist import plot_gradient_histograms
from tflux.dtypes import Sample, Junction, GridFFT, Grid, Mesh, LinReg


# TODO: figure out what this was for
def prepare_io():
    return


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
def fft_to_mesh(grid: Grid):
    if grid.grid_type == 'fourier':        
        z2_mesh = grid.z_tilde
        q_mesh, w_mesh = np.meshgrid(grid.q, grid.w, indexing='ij')
        mesh = Mesh(q_mesh, w_mesh, z2_mesh, False)
        return mesh


# Converting Grid to LinReg
def linreg_on_fft(grid: Grid):
    if grid.grid_type == 'fourier':
        linreg_q = grid.grid_to_linreg_over('q')
        linreg_w = grid.grid_to_linreg_over('w')
        return linreg_q, linreg_w


# Preprocessing Junction into Grid and Mesh
def process_surface(junc: Junction) -> Junction:
    
    junc.grid = grid_utils.grid_xt(junc)  # Constructs the Grid object
    # print(f'Grid size x: {len(junc.grid.x)}, t: {len(junc.grid.t)}')
    junc.grid = grid_utils.interpolate_zeros(junc.grid)
    junc.grid = grid_utils.trim_grid(junc.grid, crop_percent=config.CROP_PERCENT)
    # print(f'Trimmed x: {len(junc.grid.x)}, t: {len(junc.grid.t)}')

    junc.fft = junc.grid.fourier_transform(shift_fft=True, square_fft=True)
    # print(f'Trimmed fft q: {len(junc.fft.q)}, w: {len(junc.fft.w)}')
    junc.linreg_q, junc.linreg_w = linreg_on_fft(junc.fft)
    
    junc.mesh = fft_to_mesh(junc.fft)  # Contruct the Mesh object

    junc.mesh = junc.mesh.apply_masks(denoise=True)  # Slice above positive frequency and below noise floor
    junc.mesh = junc.mesh.find_loglog_gradient()
    
    return junc


### Batch Processing via Directory ###
def process_files(data_dir_path=None):

    sample = Sample()
    data_dir_path = Path(data_dir_path)

    # Find .obj files
    obj_files = sorted(data_dir_path.glob("*.obj"))
    if not obj_files:
        print(f"No OBJ files found in directory: {data_dir_path}")
        return None

    for file_path in obj_files:
        print(f"Processing file: {file_path.name}")

        top_junc, bot_junc = prepare_obj(str(file_path))  # prepare_obj expects a string path

        top_junc_processed = process_surface(top_junc)
        bot_junc_processed = process_surface(bot_junc)

        top_junc_processed.source_file = file_path
        bot_junc_processed.source_file = file_path

        sample = sample.append_junction(top_junc_processed)
        sample = sample.append_junction(bot_junc_processed)

    return sample


### PIPELINE START ###
def run_pipeline(data_dir_path: Path = None, output_dir_path: Path = None) -> None:

    # Prepare IO directories (if needed)
    if data_dir_path is None:
        data_dir_path = paths.get_data_dir()
    
    if output_dir_path is None:
        output_dir_path = paths.make_output_dir() # creates junction_summaries subdirectory as well

    # TODO: verify subdirectories exist, if not create them
    # Assume output_dir has been created with subdirectory junction_summaries

    junction_summary_dir = output_dir_path / "junction_summaries"

    # Process files and extract slopes
    sample = process_files(data_dir_path)

    slope_analyzer.save_slopes_to_csv(sample, output_dir=output_dir_path)    # Data saved to slopes.csv in output_dir_path

    if config.print_average_slopes:
        slope_analyzer.average_sample_slopes(sample, slopes=None, output_dir=output_dir_path)
    
    if config.make_histograms:
        csv_path = output_dir_path / "slopes.csv"
        fig = plot_gradient_histograms(csv_path=csv_path)
        fig.savefig(output_dir_path / 'slope_histograms.png')

    if config.make_junc_summary:
        for junc in sample.juncs:
            fig = plot_junction_summary_3x3(junc=junc)  # TODO: fix bug and verify fft plots are correct
            png_name = f'{junc.source_file.stem}_{"top" if junc.is_top else "bottom"}_3x3summary.png'
            fig.savefig(junction_summary_dir / f'{png_name}')
            plt.close(fig)
            
    return