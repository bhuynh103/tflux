# -*- coding: utf-8 -*-
### TODO: omega scaling distribution

"""
Created on Wed Dec 11 00:33:25 2024

@author: Brain Huynh
"""

import config
# import preprocessing
import analysis
import visualization
import figures
import figures2
import pandas as pd
import matplotlib.pyplot as plt
import time


def analyze_sample(directory):

    sample = analysis.process_files(directory)
    
    if config.find_average_metrics:
        metrics = ['a', 'b', 'q_m', 'w_m']
        analysis.calc_sample_metrics(sample, metrics)
        
    if config.include_visualizations:   
        for junc in sample.juncs:
            # 3 x 3 summary subplots
            fig, axs = plt.subplots(3, 3, figsize=(11, 11), squeeze=True)
            axs_flat = axs.flatten()
            
            axs_flat[0] = visualization.plot_vertices_3d(junc.vertices, 
                                           cmap=config.cmap1,
                                           title='a',
                                           ax=axs_flat[0])            
            
            axs_flat[1] = visualization.plot_xt_surface(junc.grid,
                                          cmap=config.cmap1,
                                          ax=axs_flat[1])
            
            # visualization.plot_amplitude_distribution(junc.grid,
            #                                           bins=50,
            #                                           cmap=config.cmap1,
            #                                           ax=axs[0, 2])
            
            axs_flat[2] = visualization.plot_3d_fft(junc.mesh, 
                                                     log=True, 
                                                     log_residuals=False,
                                                     include_best_fit=True, 
                                                     ax=axs_flat[2])
            
            axs_flat[3], axs_flat[6] = visualization.plot_fft_vs_q_omega(junc.grid.z_tilde, 
                                             ax1=axs_flat[3],
                                             ax2=axs_flat[6]) 
            
            visualization.plot_2d_fft_slope(junc.linreg_w, ax=axs_flat[3])
            visualization.plot_2d_fft_slope(junc.linreg_q, ax=axs_flat[6])
            
            
            axs_flat[4], axs_flat[7] = visualization.plot_fft_vs_q_omega(junc.grid.z_tilde, 
                                             ax1=axs_flat[4],
                                             ax2=axs_flat[7],
                                             scale='log') 
            
            axs_flat[5] = visualization.plot_2d_fft_slope(junc.linreg_w, ax=axs_flat[5], scale='log')
            axs_flat[8] = visualization.plot_2d_fft_slope(junc.linreg_q, ax=axs_flat[8], scale='log')
            
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
            
            # visualization.plot_fft_vs_q_omega(data["fft"], 
            #                                  ax1=axs2_flat[0],
            #                                  ax2=axs2_flat[2]) 
            
            # visualization.plot_fft_vs_q_omega(data["fft"], 
            #                                  ax1=axs2_flat[1],
            #                                  ax2=axs2_flat[3],
            #                                  scale='log') 
            
            # visualization.plot_2d_fft_slope_time(data["linreg"]["time"], ax=axs2_flat[1])
            # visualization.plot_2d_fft_slope(data["linreg"]["space"], ax=axs2_flat[3])
            
            axs2_flat[0] = visualization.plot_3d_fft(junc.mesh, 
                                                     log=True, 
                                                     log_residuals=False,
                                                     include_best_fit=True, 
                                                     ax=axs2_flat[0]) 
            
            axs2_flat[1] = visualization.plot_3d_fft(junc.mesh, 
                                                     log=True, 
                                                     log_residuals=True,
                                                     include_best_fit=True, 
                                                     ax=axs2_flat[1]) 
            
            axs2_flat[2] = visualization.plot_3d_fft(junc.mesh, 
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
    '''
    Parameters
    ----------
    directory : str
        File path to directory containing treatment directories.
        Each treatment directory should contain an experimental and control directory
        with the corresponding .obj files.

    Returns
    -------
    loglog_df : DataFrame
        "log_x": float,
        "log_y": float,
        "treatment": "Bleb" or "LatB",
        "group": "control" or "experimental",
        "junction": int,
        "sample": str
        
    alpha_df : DataFrame
        "alpha": "float",
        "treatment": "Bleb" or "LatB",
        "group": "control" or "experimental",
        "sample": str
        
    '''
    
    loglog_df, alpha_df = analysis.pair_process_directories(config.PAIR_DIRECTORY)
    
    if config.include_figures:
        figures2.fft_scatterplot(loglog_df, treatment_type="Bleb")
        figures2.fft_scatterplot(loglog_df, treatment_type="LatB")
        
        figures2.fft_alphaplot(alpha_df, treatment_type="Bleb")
        figures2.fft_alphaplot(alpha_df, treatment_type="LatB")
    
    return loglog_df, alpha_df


def main():
    '''
    Returns
    -------
    batch_plot_df : DataFrame
        See above.
    alpha_df : TYPE
        See above.

    '''
    sample = None
    # batch_plot_df = None
    # alpha_df = None
    # t_tests = None
    
    if config.process_sample_directory:
        sample = analyze_sample(config.SINGLE_DIRECTORY)
    
    # if config.process_batch_directory:
    #     batch_plot_df, alpha_df = pair_analyze_samples(config.PAIR_DIRECTORY)
        
    #     t_tests = []
    #     bleb_t_test = figures2.test_mean_alpha(alpha_df, "Bleb")
    #     latb_t_test = figures2.test_mean_alpha(alpha_df, "LatB")
    #     t_tests = [bleb_t_test, latb_t_test]
    #     bleb_t_test.to_csv('t_test_results.csv', mode='w')
    #     latb_t_test.to_csv('t_test_results.csv', mode='a')
    
    return sample # N surfaces

# Main Execution 
if __name__ == "__main__":
    start = time.process_time()
    sample = main()
    end = time.process_time()
    print(end-start)