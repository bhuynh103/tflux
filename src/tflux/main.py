# -*- coding: utf-8 -*-
### TODO: omega scaling distribution

"""
Created on Wed Dec 11 00:33:25 2024

@author: Brain Huynh
"""
from codetiming import Timer
from tflux.utils.logging import get_logger

from pathlib import Path
from tflux.pipeline import config, run
from tflux.io import paths
from tflux.plotting.sample_slope_hist import plot_all_gradient_histograms

logger = get_logger(__name__)

def main():

    # Default output path: tflux/outputs/<date_and_index>/
    # If data_dir is None, check tflux/data/raw/temp/WT as default for .obj files, otherwise error
    logger.info("Preparing I/O paths...")
    data_dir_path, output_dir_path = paths.prepare_io(
        set_data_dir_path=config.data_dir_path, 
        set_output_dir_path=None
    )

    logger.info(f"  Data dir:   {data_dir_path}")
    logger.info(f"  Output dir: {output_dir_path}")
    response = input("Proceed? [y/N]: ").strip().lower()
    if response != "y":
        logger.info("Aborted.")
        return 0

    # Save config to .txt
    if config.save_config:
        with open(Path("src/tflux/pipeline/config.py"), "r") as src, open(output_dir_path / "config.txt", "w") as out:
            out.writelines(src.readlines())
    
    # Using this function requires preprocessed csv files, running pipeline is not needed.
    if config.remake_histograms:
        plot_all_gradient_histograms(csv_path_list=config.CSV_PATHS, output_dir=output_dir_path)
        return 0
    
    # Process sample from data dir, create metrics.csv, junction summaries, and histograms
    with Timer(text="Pipeline: {:.3f}s", logger=logger.info):
        run.run_pipeline(data_dir_path=data_dir_path, output_dir_path=output_dir_path, sample_label="WT_temp")   # Save slope data to metrics.csv
    
    return 0

# Main Execution 
if __name__ == "__main__":
    main()
