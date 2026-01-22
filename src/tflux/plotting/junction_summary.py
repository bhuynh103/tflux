import matplotlib.pyplot as plt
import tflux.pipeline.config as config
import tflux.plotting.visualization as vis

def plot_junction_summary(junc):
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
    # plt.show()
    
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
    # plt.show()