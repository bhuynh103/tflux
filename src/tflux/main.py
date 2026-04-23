# -*- coding: utf-8 -*-
### TODO: omega scaling distribution

"""
Created on Wed Dec 11 00:33:25 2024

@author: Brain Huynh
"""
import matplotlib.pyplot as plt
from codetiming import Timer
from tflux.utils.logging import get_logger

from pathlib import Path
from tflux.pipeline import config, run
from tflux.io import paths
from tflux.plotting.sample_slope_hist import compare_linreg_fits, compare_linreg_hists

logger = get_logger(__name__)

def main():
    "TODO: Add CLI function to clear outputs folder"
    logger.info("Preparing I/O paths...")

    # Define your data sources
    my_data = {
        "WT": Path(config.sample_WT_dir_path),
        "Control": Path(config.sample_a_dir_path),
        "Experimental": Path(config.sample_b_dir_path)
    }

    # Initialize IO
    input_dirs, output_dir = paths.prepare_io(data_paths=my_data)
    response = input("Proceed? [y/N]: ").strip().lower()
    if response != "y":
        logger.info("Aborted.")
        return 0

    # Save config to .txt
    if config.save_config:
        with open(Path("src/tflux/pipeline/config.py"), "r") as src, open(output_dir / "config.txt", "w") as out:
            out.writelines(src.readlines())
    
    
    # Process sample from data dir, create metrics.csv, junction summaries, and histograms
    with Timer(text="Pipeline: {:.3f}s", logger=logger.info):
        sample_WT = run.run_pipeline(data_dir_path=input_dirs["WT"], output_dir_path=output_dir, sample_label="WT")

        if config.compare_samples_ab:
            sample_a = run.run_pipeline(data_dir_path=input_dirs["Control"], output_dir_path=output_dir, sample_label="control")
            sample_b = run.run_pipeline(data_dir_path=input_dirs["Experimental"], output_dir_path=output_dir, sample_label="experimental")

            comparison_dir = output_dir / "comparisons"

            fig_hist = compare_linreg_hists(sample_a=sample_a, sample_b=sample_b)
            png_name = f'hist_comparison.png'
            fig_hist.savefig(comparison_dir / png_name)
            plt.close(fig_hist)

            fig_fit = compare_linreg_fits(sample_a=sample_a, sample_b=sample_b)
            png_name = f'fit_comparison.png'
            fig_fit.savefig(comparison_dir / png_name)
            plt.close(fig_fit)

    
    return 0

# Main Execution 
if __name__ == "__main__":
    main()
