# -*- coding: utf-8 -*-
### TODO: omega scaling distribution

"""
Created on Wed Dec 11 00:33:25 2024

@author: Brain Huynh
"""

from tflux.pipeline import config, run
from tflux.io import paths
from tflux.plotting import sample_slope_hist
import matplotlib.pyplot as plt


def main():
    if config.process_sample_directory:
        run.run_pipeline(data_dir_path=config.DATA_DIR_PATH)   # Save slope data to metrics.csv

    if config.make_histograms:

        root = paths.find_root()
        csv_path = root / config.CSV_PATH
        fig, axes = sample_slope_hist.plot_gradient_histograms(csv_path=csv_path)
        plt.show()
    return

# Main Execution 
if __name__ == "__main__":
    main()
