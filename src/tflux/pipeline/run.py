import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tflux.pipeline.config as config
import tflux.preprocessing.grid_utils as grid_utils
import tflux.preprocessing.vertices_utils as vertices_utils
import tflux.io.obj_reader as obj_reader
import tflux.analysis.metrics as metrics_analyzer
import tflux.plotting.visualization as vis
import tflux.plotting.figures2 as fig2
from tflux.dtypes import Sample, Junction, Grid, Mesh, LinReg

# Preprocessing .obj into top and bottom Junctions
def prepare_obj(file):

    vertices = obj_reader.load_obj_vertices(file)
        
    best_angle, best_vertices = vertices_utils.find_best_orientation(vertices)
        
    best_vertices[:, 0] *= config.dt # pixels to seconds
    best_vertices[:, 1:] *= config.dx # pixels to meters

    top_half, bottom_half = vertices_utils.slice_vertices(best_vertices)
    # top_half_centralized = centralize_vertices(top_half) # Introduces discontinuity
    # bottom_half_centralized = centralize_vertices(bottom_half)
    
    top_junc = Junction(vertices=top_half, is_top=True, filename=file)
    bot_junc = Junction(vertices=bottom_half, is_top=False, filename=file)
    
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
def process_files(directory_path=None):
    """
    Reads .obj files from <project_root>/data/raw/all-data/WT by default.

    project_root is inferred as the directory containing both 'src' and 'data'.
    """
    # --- infer project root from this file's location ---
    here = Path(__file__).resolve()
    print(here)

    # Walk upward until we find a folder that has both 'src' and 'data'
    project_root = None
    for p in [here.parent, *here.parents]:
        if (p / "src").is_dir() and (p / "data").is_dir():
            project_root = p
            break
    if project_root is None:
        # Fallback: assume repo root is two levels above src/tflux/...
        project_root = here.parents[2]

    # --- default data directory ---
    default_data_dir = project_root / "data" / "raw" / "all-data" / "WT"

    # allow override, but default to WT folder
    data_dir = Path(directory_path).resolve() if directory_path else default_data_dir

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    sample = Sample()

    # Find .obj files
    obj_files = sorted(data_dir.glob("*.obj"))
    if not obj_files:
        print(f"No OBJ files found in directory: {data_dir}")
        return None

    for file_path in obj_files:
        file_basename = file_path.name
        print(f"\nProcessing file: {file_basename}")

        top_junc, bot_junc = prepare_obj(str(file_path))  # if prepare_obj expects a string path

        top_junc_processed = process_surface(top_junc)
        bot_junc_processed = process_surface(bot_junc)

        sample = sample.append_junction(top_junc_processed)
        sample = sample.append_junction(bot_junc_processed)

    return sample


### PIPELINE START ###
def run_pipeline(directory_path=None):

    sample = process_files(directory_path)
    
    # analyze_sample(sample)
    if config.find_average_metrics:
        metrics = ['a', 'b', 'q_m', 'w_m']
        metrics_analyzer.calc_sample_metrics(sample, metrics)
    
    # visualize_sample(sample)
    if config.include_visualizations:   
        for junc in sample.juncs:
            # 3 x 3 summary subplots
            fig, axs = plt.subplots(3, 3, figsize=(11, 11), squeeze=True)
            axs_flat = axs.flatten()
            
            axs_flat[0] = vis.plot_vertices_3d(junc.vertices, 
                                           cmap=config.cmap1,
                                           title='a',
                                           ax=axs_flat[0])            
            
            axs_flat[1] = vis.plot_xt_surface(junc.grid,
                                          cmap=config.cmap1,
                                          ax=axs_flat[1])
            
            # vis.plot_amplitude_distribution(junc.grid,
            #                                           bins=50,
            #                                           cmap=config.cmap1,
            #                                           ax=axs[0, 2])
            
            axs_flat[2] = vis.plot_3d_fft(junc.mesh, 
                                                     log=True, 
                                                     log_residuals=False,
                                                     include_best_fit=True, 
                                                     ax=axs_flat[2])
            
            axs_flat[3], axs_flat[6] = vis.plot_fft_vs_q_omega(junc.grid.z_tilde, 
                                             ax1=axs_flat[3],
                                             ax2=axs_flat[6]) 
            
            vis.plot_2d_fft_slope(junc.linreg_w, ax=axs_flat[3])
            vis.plot_2d_fft_slope(junc.linreg_q, ax=axs_flat[6])
            
            
            axs_flat[4], axs_flat[7] = vis.plot_fft_vs_q_omega(junc.grid.z_tilde, 
                                             ax1=axs_flat[4],
                                             ax2=axs_flat[7],
                                             scale='log') 
            
            axs_flat[5] = vis.plot_2d_fft_slope(junc.linreg_w, ax=axs_flat[5], scale='log')
            axs_flat[8] = vis.plot_2d_fft_slope(junc.linreg_q, ax=axs_flat[8], scale='log')
            
            letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
            for ax, l in zip(axs.flatten(), letters):
                ax.set_title(f'{l}', y=1.05)
            fig.suptitle(f'{junc}')
            
            plt.subplots_adjust(right=1.1, wspace=0.7, hspace=0.7)
            plt.show()
            
            ######################
            # 2 x 2 FFT summary subplots
            fig2, axs2 = plt.subplots(2, 2, figsize=(11, 11), squeeze=True) # sharey = 'row'
            axs2_flat = axs2.flatten()
            
            # vis.plot_fft_vs_q_omega(data["fft"], 
            #                                  ax1=axs2_flat[0],
            #                                  ax2=axs2_flat[2]) 
            
            # vis.plot_fft_vs_q_omega(data["fft"], 
            #                                  ax1=axs2_flat[1],
            #                                  ax2=axs2_flat[3],
            #                                  scale='log') 
            
            # vis.plot_2d_fft_slope_time(data["linreg"]["time"], ax=axs2_flat[1])
            # vis.plot_2d_fft_slope(data["linreg"]["space"], ax=axs2_flat[3])
            
            axs2_flat[0] = vis.plot_3d_fft(junc.mesh, 
                                                     log=True, 
                                                     log_residuals=False,
                                                     include_best_fit=True, 
                                                     ax=axs2_flat[0]) 
            
            axs2_flat[1] = vis.plot_3d_fft(junc.mesh, 
                                                     log=True, 
                                                     log_residuals=True,
                                                     include_best_fit=True, 
                                                     ax=axs2_flat[1]) 
            
            axs2_flat[2] = vis.plot_3d_fft(junc.mesh, 
                                                     log=False, 
                                                     log_residuals=False,
                                                     include_best_fit=False, 
                                                     ax=axs2_flat[2]) 
            
            
            letters = ['e', 'f', 'h', 'i']
            for ax, l in zip(axs2.flatten(), letters):
                ax.set_title(f'{l}')
            fig.suptitle(f'{junc}')
            
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            plt.show()
            
    return sample # N surfaces


def pair_analyze_samples(directory):
    
    loglog_df, alpha_df = analysis.pair_process_directories(config.PAIR_DIRECTORY)
    
    if config.include_figures:
        fig2.fft_scatterplot(loglog_df, treatment_type="Bleb")
        fig2.fft_scatterplot(loglog_df, treatment_type="LatB")
        
        fig2.fft_alphaplot(alpha_df, treatment_type="Bleb")
        fig2.fft_alphaplot(alpha_df, treatment_type="LatB")
    
    return loglog_df, alpha_df