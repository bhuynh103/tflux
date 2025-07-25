# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 00:33:25 2024

@author: Brain Huynh
"""


import config
# import preprocessing
import analysis
import visualization


def main() -> tuple:
    
    results, grid_size, N = analysis.process_files_in_directory(config.DIRECTORY, 
                                                             config.dx, 
                                                             config.dt)
    
    if config.include_visualizations:   
        for index, data in enumerate(results.values(), start=1):
            
            visualization.plot_vertices_3d(data["vertices"], 
                                           index)

            visualization.plot_xt_surface(data["xt_surface"], 
                                          data["t_range"], 
                                          data["x_range"], 
                                          index)
            
            visualization.plot_3d_fft_loglog(data["fft_shifted"], 
                                             data["t_range"], 
                                             data["x_range"], 
                                             index)
            
            visualization.plot_2d_fft_slope(data["linreg"], 
                                            index)
    
    metrics = {}
    
    if config.find_average_metrics:
        slope_list = [data["slope"] for data in results.values()]
        intercept_list = [data["intercept"] for data in results.values()]
        tension_list = [data["tension"] for data in results.values()]
            
        metrics_mean, metrics_std_err = analysis.calculate_metrics(slope_list, intercept_list, tension_list)
        
        print(metrics_mean, metrics_std_err)
        metrics = {"slope": {"mean": metrics_mean[0],
                             "std_err": metrics_std_err[0]},
                   "intercept": {"mean": metrics_mean[1],
                             "std_err": metrics_std_err[1]},
                   "tension": {"mean": metrics_mean[2],
                               "std_err": metrics_std_err[2]},}
        
    return results, metrics, N

# Main Execution 
if __name__ == "__main__":
    results, metrics, N = main()