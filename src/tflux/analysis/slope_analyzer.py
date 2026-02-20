# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:44:22 2025

@author: bhuyn
"""

import csv
from pathlib import Path
import tflux.pipeline.config as config       


# Stats

def average_sample_slopes(sample, slopes: list[str], output_dir=None):
    """
    Calculate and save average slopes to a text file.
    """
    if slopes is None:
        slopes = ['a', 'b', 'q_m', 'w_m']

    N = len(sample.juncs)
    if N >= 1:
        # Prepare output file
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_file = output_dir / "slopes.txt"
        else:
            output_file = "slopes.txt"
        
        # Calculate slopes and write to file
        with open(output_file, 'w') as f:
            f.write(f"Sample slopes (N = {N} junctions)\n")
            f.write("=" * 50 + "\n\n")
            
            for metric in slopes:
                mean, std = sample.find_average_metric(metric)
                line = f'{metric} = {mean} +/- {std}\n'
                print(line.rstrip())  # Print to console
                f.write(line)  # Write to file
        
        print(f"\nslopes saved to: {output_file}")


def save_slopes_to_csv(sample, output_dir=None, filename="slopes.csv"):
    """
    Save all junction slopes to a CSV file.
    """
    if len(sample.juncs) == 0:
        print("No junctions to save.")
        return
    
    # Prepare output file path
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_file = output_dir / filename
    else:
        output_file = filename
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['grad_q', 'grad_w', 'linreg_q', 'linreg_w', 'source'])
        
        # Write data rows
        for junc in sample.juncs:
            writer.writerow([junc.mesh.a, junc.mesh.b, junc.linreg_q.m, junc.linreg_w.m, junc.source_file])
    
    print(f"Saved {len(sample.juncs)} junctions to: {output_file}")
    return output_file


def tension_interpolation(interp):
    return (config.boltzmann_constant * config.room_temp) / ((10 ** (interp + 4.5)))




# def pair_process_directories(directory):  
#     treatment_directory_list = get_directories_in_path(directory)
#     scatterplot_dfs = []
#     alpha_dfs = []
    
#     for treatment_directory in treatment_directory_list:
#         group_list = get_directories_in_path(treatment_directory)
        
#         for group_index, group_dir_path in enumerate(group_list):
#             sample = process_files(group_dir_path)
#             slope_array_positive = np.array(slope_list) * -1
            
#             for junction_index, linreg in enumerate(linreg_list):
#                 x = linreg["log_x"]
#                 y = linreg["log_y"]
#                 sample = np.array([f"{group_dir_path}" for i in range(0, len(x))])
#                 treatment = np.array([f"{os.path.basename(treatment_directory)}" for i in range(0, len(x))])
#                 junction = np.array([junction_index for i in range(0, len(x))])
#                 group_index_list = np.array([group_index for i in range(0, len(x))])
#                 group_labels = np.where(group_index_list == 0, "control", "experimental")
#                 scatterplot_df_chunk = pd.DataFrame(data=list(zip(x, y, treatment, group_labels, junction, sample)), columns=["log_x", "log_y", "treatment", "group", "junction", "sample"])
#                 scatterplot_dfs.append(scatterplot_df_chunk)
            
#             sample_alpha = np.array([f"{group_dir_path}" for i in range(0, len(slope_list))])
#             treatment_alpha = np.array([f"{os.path.basename(treatment_directory)}" for i in range(0, len(slope_list))])
#             group_index_list_alpha = np.array([group_index for i in range(0, len(slope_list))])
#             group_label_alpha = np.where(group_index_list_alpha == 0, "control", "experimental")
#             alpha_df_chunk = pd.DataFrame(data=list(zip(slope_array_positive, treatment_alpha, group_label_alpha, sample_alpha)), columns=["alpha", "treatment", "group", "sample"])
#             alpha_dfs.append(alpha_df_chunk)
    
#     scatterplot_df = pd.concat(scatterplot_dfs, ignore_index=True)
#     scatterplot_df.to_csv("paired_figure_data.csv", index=False)
    
#     alpha_df = pd.concat(alpha_dfs, ignore_index=True)
#     alpha_df.to_csv("paired_alpha_data.csv", index=False)

#     return scatterplot_df, alpha_df
