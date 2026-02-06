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
    run.run_pipeline(data_dir_path=config.DATA_DIR_PATH)   # Save slope data to metrics.csv
    return

# Main Execution 
if __name__ == "__main__":
    main()
