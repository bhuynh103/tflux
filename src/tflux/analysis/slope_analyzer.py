# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:44:22 2025

@author: bhuyn
"""

import csv
from pathlib import Path
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


def tension_interpolation(interp):
    return (config.boltzmann_constant * config.room_temp) / ((10 ** (interp + 4.5)))

