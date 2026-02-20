# -*- coding: utf-8 -*-
### TODO: omega scaling distribution

"""
Created on Wed Dec 11 00:33:25 2024

@author: Brain Huynh
"""

from tflux.pipeline import config, run
from tflux.io import paths
import matplotlib.pyplot as plt


def main():
    if config.process_sample_directory:
        run.run_pipeline(data_dir_path=config.DATA_DIR_PATH)   # Save slope data to metrics.csv

    if config.make_histograms:
        root = paths.find_root()
        # csv_path = root / config.CSV_PATH
        for csv_path in config.CSV_PATHS:
            csv_path = root / csv_path
            fig, axes = sample_slope_hist.plot_gradient_histograms(csv_path=csv_path, bins=16)
            fig.set_subplot_titles([
                r'From Gradient: $\frac{\partial u^2}{\partial q}$',
                r'From Gradient: $\frac{\partial u^2}{\partial \omega}$', 
                r'From Averaging: $\frac{d \langle u^2 \rangle_\omega}{d q}$',
                r'From Averaging: $\frac{d \langle u^2 \rangle_q}{d \omega}$'
            ])
            fig.set_subplot_xlabels([
                r'$\alpha$',
                r'$\alpha$',
                r'$\alpha$', 
                r'$\alpha$'
            ])
            plt.show()

    return

# Main Execution 
if __name__ == "__main__":
    main()
