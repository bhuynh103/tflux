import numpy as np
import os
import config
import preprocessing as pp
import io
from types import Sample, Junction, Grid, Mesh, LinReg

# Preprocessing .obj into top and bottom Junctions
def prepare_obj(file):

    vertices = io.load_obj_vertices(file)
        
    best_angle, best_vertices = pp.vertices.find_best_orientation(vertices)
        
    best_vertices[:, 0] *= config.dt # pixels to seconds
    best_vertices[:, 1:] *= config.dx # pixels to meters

    top_half, bottom_half = pp.vertices.slice_vertices(best_vertices)
    # top_half_centralized = centralize_vertices(top_half) # Introduces discontinuity
    # bottom_half_centralized = centralize_vertices(bottom_half)
    
    top_junc = Junction(vertices=top_half, is_top=True, filename=file)
    bot_junc = Junction(vertices=bottom_half, is_top=False, filename=file)
    
    return top_junc, bot_junc


# Preprocessing Junction into Grid and Mesh
def process_surface(junc: Junction):
    # results = {}
    
    junc = pp.grid_utils.grid_xt(junc)  # Construct the Grid object
    junc.grid = pp.grid_utils.interpolate_zeros(junc.grid)
    # junc.grid = pp.grid_utils.trim_grid(junc.grid)
    junc.grid = junc.grid.fourier_transform(shift_fft=True, square_fft=True) # Method
    print(junc.grid.shifted, junc.grid.squared)
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
def process_files(directory_path):
    # results = {}
    sample = Sample()
    obj_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.obj')]

    if not obj_files:
        print(f"No OBJ files found in directory: {directory_path}")
        return

    for file in obj_files:
        file_basename = os.path.basename(file)
        print(f"\nProcessing file: {file_basename}")
        
        top_junc, bot_junc = prepare_obj(file)  # Pixels to seconds and meters
        # print(f"  Cell Length: {x_range * 10 ** 6:2f} microns")
        
        # results[f"top:{file_basename}"] = process_surface(top_junc)
        # results[f"bot:{file_basename}"] = process_surface(bot_junc)
        
        top_junc_processed = process_surface(top_junc)
        bot_junc_processed = process_surface(bot_junc)
        
        sample = sample.append_junction(top_junc_processed)
        sample = sample.append_junction(bot_junc_processed)
    
    # N = len(results)
    
    # results_df = pd.DataFrame(columns=["linreg_q_slope", "linreg_q_int", "mesh_a", "mesh_b"])
    
    # linreg_list = [junc.linreg for junc in results.values()]
    # slope_list = [junc.linreg.m for junc in results.values()]
    # intercept_list = [junc.linreg.b for junc in results.values()]
    # # tension_list = [data["tension"] for data in results.values()]
    # a_list = [junc.mesh.a for junc in results.values()]
    # b_list = [junc.mesh.b for junc in results.values()]
    
    return sample


