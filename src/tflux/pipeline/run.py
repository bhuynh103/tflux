from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tflux.io.paths as paths
import tflux.pipeline.config as config
import tflux.preprocessing.grid_utils as grid_utils
import tflux.preprocessing.vertices_utils as vertices_utils
import tflux.io.obj_reader as obj_reader
import tflux.analysis.metrics as metrics_analyzer
from tflux.plotting.junction_summary import plot_junction_summary
import tflux.plotting.figures2 as fig2
from tflux.dtypes import Sample, Junction, Grid, Mesh, LinReg


def prepare_io():
    return


# Preprocessing .obj into top and bottom Junctions
def prepare_obj(file):

    vertices = obj_reader.load_obj_vertices(file)
        
    best_angle, best_vertices = vertices_utils.find_best_orientation(vertices)
        
    best_vertices[:, 0] *= config.dt # pixels to seconds
    best_vertices[:, 1:] *= config.dx # pixels to meters

    top_half, bottom_half = vertices_utils.slice_vertices(best_vertices)
    # top_half_centralized = centralize_vertices(top_half) # Introduces discontinuity
    # bottom_half_centralized = centralize_vertices(bottom_half)
    
    top_junc = Junction(vertices=top_half, is_top=True)
    bot_junc = Junction(vertices=bottom_half, is_top=False)
    
    return top_junc, bot_junc


# Preprocessing Junction into Grid and Mesh
def process_surface(junc: Junction):
    # results = {}
    
    junc = grid_utils.grid_xt(junc)  # Construct the Grid object
    junc.grid = grid_utils.interpolate_zeros(junc.grid)
    # junc.grid = grid_utils.trim_grid(junc.grid)
    junc.grid = junc.grid.fourier_transform(shift_fft=True, square_fft=True) # Method
    junc.linreg_q, junc.linreg_w = linreg_on_fft(junc.grid)
    
    junc.mesh = fft_to_mesh(junc.grid)  # Contruct the Mesh object

    junc.mesh = junc.mesh.apply_masks(denoise=True)  # Slice above positive frequency and below noise floor
    junc.mesh = junc.mesh.find_loglog_gradient()
    
    return junc


# Converting Grid to Mesh
def fft_to_mesh(grid: Grid):
    if grid.grid_type == 'fourier':        
        z2_mesh = grid.z_tilde
        z2_mesh_transpose = z2_mesh.T
        q_mesh, w_mesh = np.meshgrid(grid.q, grid.w, indexing='ij')
        mesh = Mesh(q_mesh, w_mesh, z2_mesh_transpose, False)
        return mesh


# Converting Grid to LinReg
def linreg_on_fft(grid: Grid):
    if grid.grid_type == 'fourier':
        linreg_q = grid.grid_to_linreg_over('q')
        linreg_w = grid.grid_to_linreg_over('w')
        return linreg_q, linreg_w


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
        file_basename = file_path.name
        print(f"\nProcessing file: {file_basename}")

        top_junc, bot_junc = prepare_obj(str(file_path))  # if prepare_obj expects a string path

        top_junc_processed = process_surface(top_junc)
        bot_junc_processed = process_surface(bot_junc)

        top_junc_processed.source_file = file_basename
        bot_junc_processed.source_file = file_basename

        sample = sample.append_junction(top_junc_processed)
        sample = sample.append_junction(bot_junc_processed)

    return sample


### PIPELINE START ###
def run_pipeline(data_dir_path=None):

    if data_dir_path is None:
        data_dir_path = paths.get_data_dir()
    
    output_dir_path = paths.make_output_dir()

    sample = process_files(data_dir_path)

    metrics_analyzer.save_metrics_to_csv(sample, output_dir=output_dir_path)    # Data saved

    if config.find_average_slopes:
        metrics = ['a', 'b', 'q_m', 'w_m']
        metrics_analyzer.average_sample_metrics(sample, metrics, output_dir=output_dir_path)
    
    if config.include_junc_summary:
        for junc in sample.juncs:
            plot_junction_summary(junc=junc)
            
    return