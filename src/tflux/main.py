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
from tflux.analysis.slope_analyzer import ttest_linreg_slopes

logger = get_logger(__name__)

def main():
    logger.info("Preparing I/O paths...")

    my_data = {
        "WT": Path(config.sample_WT_dir_path),
        "control": Path(config.sample_a_dir_path),
        "experimental": Path(config.sample_b_dir_path)
    }
    input_dirs, output_dir = paths.prepare_io(data_paths=my_data)
    
    response = input("Proceed? [y/N]: ").strip().lower()
    if response != "y":
        logger.info("Aborted.")
        return 0

    if config.save_config:
        with open(Path("src/tflux/pipeline/config.py"), "r") as src, open(output_dir / "config.txt", "w") as out:
            out.writelines(src.readlines())
    
    # Start pipeline -> Analyze sample WT -> Compare samples A and B
    with Timer(text="Pipeline: {:.3f}s", logger=logger.info):
        sample_WT = run.run_pipeline(data_dir_path=input_dirs["WT"], output_dir_path=output_dir, sample_label="WT")

        if config.compare_samples_ab:
            sample_a = run.run_pipeline(data_dir_path=input_dirs["control"], output_dir_path=output_dir, sample_label="control")
            sample_b = run.run_pipeline(data_dir_path=input_dirs["experimental"], output_dir_path=output_dir, sample_label="experimental")

            comparison_dir = output_dir / "comparisons"

            fig_hist = compare_linreg_hists(sample_a=sample_a, sample_b=sample_b)
            png_name = f'hist_comparison.png'
            fig_hist.savefig(comparison_dir / png_name)
            plt.close(fig_hist)

            fig_fit = compare_linreg_fits(sample_a=sample_a, sample_b=sample_b)
            png_name = f'fit_comparison.png'
            fig_fit.savefig(comparison_dir / png_name)
            plt.close(fig_fit)

            results = ttest_linreg_slopes(sample_a, sample_b, labels=("Bleb_control", "Bleb_experimental"))
            for dim, r in results.items():
                logger.info(
                    "dim=%s  t=%.3f  p=%.4f  significant=%s",
                    dim, r["t_stat"], r["p_value"], r["significant"]
                )


    return 0

if __name__ == "__main__":
    main()
