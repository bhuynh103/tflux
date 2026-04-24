# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:17:16 2025

@author: bhuyn
"""
from pathlib import Path
from datetime import datetime

save_config = True

### Data Paths ###
sample_WT_dir_path: Path = Path("data/raw/all-data/WT")                 # WT (isolated)
sample_a_dir_path: Path = Path("data/raw/all-data/Bleb/control")        # control (compared with b), requires 'compare_samples_ab = True' to analyze
sample_b_dir_path: Path = Path("data/raw/all-data/Bleb/experimental")   # experimental (compared with a), requires 'compare_samples_ab = True' to analyze

### Logger Settings ###
LOG_LEVEL = "INFO"          # Choose level of terminal logs, DEBUG is most verbose. [DEBUG, INFO, WARNING, ERROR, CRITICAL]
LOG_FILE = f"logs/tflux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"     # Create unique log file name on run.

### Pipeline Run ###
''' 
Each call of pipeline.run.run_pipeline() takes one directory of .obj files (or the most recent .pkl)
and performs a sequence of analyses as determined by this config.
'''
summarize_junc = False       # Generate analysis plots for every junction found in sample, Figure 1c-1k. SLOW with lots of .objs.
save_cells = False           # Save a render of each cell in the sample. Figure 1b. SLOW with lots of .objs.
save_pickle = True          # Only works when .objs are processed. Serializes Sample objects as .pkl within data directory. Overrides old .pkl ONLY IF user opts to use .objs.
make_pdfs = False            # Save junction PNGs (if generated) and cell PNGs (if generated) of samples as pdfs.
save_average_slope_csv = True      # Save average slopes of the whole sample as a CSV.
drop_bad_junctions = True          # Junctions with percent_zero > PERCENT_ZERO_THRESHOLD
make_histogram = True              # Create histogram and lineplot of slopes within sample

### Comparative Analysis ###
compare_samples_ab = True   # Run run_pipeline() on Sample A and Sample B and do comparison histogram and lineplot of slopes between samples. Figres 2+3.


### IF CHANGING ANYTHING BELOW THIS, DELETE OLD .pkl FILES FROM data FOLDER BEFORE RUNNING ###

# K-means Clustering Hyperparams
K_MEANS: int = 3
K_MEANS_ITER: int = 40  # Small effect on performance
SMOOTH_ITER: int = 3    # Big effect on performace
LAMBDA: float = 0.9
MIN_ISLAND_FACES: int = 10000   # Small effect on performance
SEED: int = 0
NORMAL_WEIGHT: float = 1.0
GEOM_WEIGHT: float = 0.5
PERCENT_ZERO_THRESHOLD: float = 0.25    # 0.25 seems to be a good filter for bad junctions

# Grid Interpolation Params
WINDOW_SIZE: int = 7            # For interpolation, screening window searches grid for sparse empty bins
MAJORITY_PERCENT: float = 0.4   # For interpolation, if window is MAJORITY_PERCENT (mostly) non-zero, then the bin is sparse
SUFFICIENT_COUNT: int = 1       # For interpolation, each bin should have SUFFICIENT_COUNT (>1) vertices, or else interpolated

# Trimming edge effects
CROP_PERCENT: float = 0.3       # Crops (CROP_PERCENT / 2) from left and right of x-range on Grid before FFT.
TANGENT_CUTOFF: float = 6       # Noise floor exists at log(q) = TANGENT_CUTOFF, omit from log-log linear regression
TANGENT_CUTOFF_TIME: float = -1        # Noise floor exists at log(omega) = TANGENT_CUTOFF_TIME, omit from log-log linear regression

# SI Units
do_scaling: bool = True         # Scale .obj to SI units upon loading
dx: float = 0.205 * (10 ** -6)  # meters per x pixel. Scales .obj units to meters
dt: float = 1.0                 # seconds per t pixel. Scales .obj units to seconds
boltzmann_constant: float = 1.36 * (10 ** -23)  # Joules per Kelvin
room_temp: float = 298                          # Kelvin