# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:44:22 2025

@author: bhuyn
"""

import csv
import numpy as np
from pathlib import Path
from scipy import stats

import tflux.pipeline.config as config
from tflux.utils.logging import get_logger
from tflux.dtypes import Sample

logger = get_logger(__name__)

def average_sample_slopes(sample, output_dir: Path):
    """
    Calculate and save average slopes to a text file.

    Inputs:
        sample: Sample      Input Sample object
        output_dir: Path    Output directory
    Returns:
        0                   Valid
    """
    # Contains which slopes to save from ['a', 'b', 'q_m', 'w_m'], defaults to saving all
    slopes = ['a', 'b', 'q_m', 'w_m']

    N = len(sample.valid_juncs)
    logger.info(f"Analyzing slopes of N = {N} valid junctions.")
    if N >= 1:
        # Prepare output file
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_file = output_dir / "slopes.txt"
        
        # Calculate slopes and write to file
        with open(output_file, 'w') as f:
            f.write(f"Sample slopes (N = {N} junctions)\n")
            f.write("=" * 50 + "\n\n")
            
            for metric in slopes:
                mean, std = sample.find_average_metric(metric)
                line = f'{metric} = {mean:.2f} +/- {std:.2f}\n'
                logger.info(line.rstrip())  # Log to console
                f.write(line)  # Write to file
        
        logger.info(f"Slopes saved to: {output_file}")

    return 0


def save_slopes_to_csv(sample: Sample, output_dir: Path):
    """
    Save all junction slopes to a CSV file.
    """
    if len(sample.valid_juncs) == 0:
        logger.warning("No junctions to save.")
        return
    
    # Prepare output file path
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_file = output_dir / 'slopes.csv'
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['grad_q', 'grad_w', 'linreg_q', 'linreg_w', 'source'])
        
        # Write data rows
        for junc in sample.valid_juncs:
            writer.writerow([junc.mesh.a, junc.mesh.b, junc.linreg_q.m, junc.linreg_w.m, junc.source_file])
    
    logger.info(f"Saved {len(sample.valid_juncs)} junctions to: {output_file}")
    return 0


def ttest_linreg_slopes(
    sample_a: Sample,
    sample_b: Sample,
    labels: tuple[str, str] = ("A", "B"),
) -> dict[str, dict]:
    """
    Perform independent two-sample t-tests on linreg slope distributions
    between two samples, for each regression dimension (q and ω).

    Parameters
    ----------
    sample_a, sample_b : Sample
        Samples whose valid junctions carry `linreg_q` and `linreg_w` attributes.
    labels : tuple[str, str]
        Display names for the two samples (used in the returned dict keys).

    Returns
    -------
    results : dict[str, dict]
        Keyed by dimension name ("q", "omega").  Each value contains:
            slopes_a, slopes_b  – raw slope arrays
            mean_a, mean_b      – sample means
            std_a,  std_b       – sample std devs
            n_a,    n_b         – sample sizes
            t_stat              – t-statistic
            p_value             – two-tailed p-value
            significant         – bool, p < 0.05
    """
    dims = [
        ("q",     "linreg_q"),
        ("omega", "linreg_w"),
    ]
    results: dict[str, dict] = {}

    for dim_name, attr in dims:
        def _slopes(sample: Sample) -> np.ndarray:
            juncs = [j for j in sample.valid_juncs if getattr(j, attr) is not None]
            return np.array([getattr(j, attr).m for j in juncs])

        slopes_a = _slopes(sample_a)
        slopes_b = _slopes(sample_b)

        t_stat, p_value = stats.ttest_ind(slopes_a, slopes_b, equal_var=False)  # Welch's t-test

        results[dim_name] = {
            f"slopes_{labels[0]}": slopes_a,
            f"slopes_{labels[1]}": slopes_b,
            f"mean_{labels[0]}":   slopes_a.mean(),
            f"mean_{labels[1]}":   slopes_b.mean(),
            f"std_{labels[0]}":    slopes_a.std(),
            f"std_{labels[1]}":    slopes_b.std(),
            f"n_{labels[0]}":      len(slopes_a),
            f"n_{labels[1]}":      len(slopes_b),
            "t_stat":              t_stat,
            "p_value":             p_value,
            "significant":         p_value < 0.05,
        }

    return results


def tension_interpolation(interp):
    return (config.boltzmann_constant * config.room_temp) / ((10 ** (interp + 4.5)))

