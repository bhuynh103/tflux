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
sample_a_dir_path: Path = Path("data/raw/all-data/LatB/control")        # control (compared with b), requires compare_samples_ab = True
sample_b_dir_path: Path = Path("data/raw/all-data/LatB/experimental")   # experimental (compared with a), requires compare_samples_ab = True

### Logger Settings ###
LOG_LEVEL = "INFO"          # Choose level of terminal logs, DEBUG is most verbose. [DEBUG, INFO, WARNING, ERROR, CRITICAL]
LOG_FILE = f"logs/tflux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"     # Create unique log file name on run.

### Pipeline Run ###
# Each call of pipeline.run.run_pipeline() takes one directory of .obj files (or the most recent .pkl) and performs a sequence of analyses as determined by this config.
summarize_junc = True     # Generate analysis plots for every junction found in sample, Figure 1c-1k. Can be slow with lots of .objs.
save_cells = True           # Save a render of each cell in the sample. Figure 1b. Can be slow with lots of .objs.
save_pickle = True          # Serialize Sample objects as .pkl within data directory. Script checks if .pkl exists, so double-check for old pickles before running.
make_pdfs = True            # Save junction PNGs (if generated) and cell PNGs (if generated) of samples as pdfs.
save_average_slope_csv = True      # Save average slopes of the whole sample as a CSV.
drop_bad_junctions = True         # Junctions with percent_zero > PERCENT_ZERO_THRESHOLD
make_histogram = True               # Create histogram and lineplot of slopes within sample

### Comparative Analysis ###
compare_samples_ab = True   # Run run_pipeline() on Sample A and Sample B and do comparison histogram and lineplot of slopes between samples. Figres 2+3.

# plotting
cmap1 = 'viridis'
cmap2 = 'spring'

### IF CHANGING ANYTHING BELOW THIS, DELETE OLD .pkl FILES FROM data FOLDER ###

# k-means hyperparams
K_MEANS: int = 3
K_MEANS_ITER: int = 40  # Small effect on performance
SMOOTH_ITER: int = 3    # Lower for better performace
LAMBDA: float = 0.9
MIN_ISLAND_FACES: int = 10000   # Small effect on performance
SEED: int = 0
NORMAL_WEIGHT: float = 1.0
GEOM_WEIGHT: float = 0.5
PERCENT_ZERO_THRESHOLD: float = 0.25

# grid interpolation hyperparams
CROP_PERCENT: float = 0.3  # crops (CROP_PERCENT / 2) from left and right
WINDOW_SIZE: int = 7
MAJORITY_PERCENT: float = 0.4
SUFFICIENT_COUNT: int = 1
MIN_VALUE = 0

# Noise floor threshold
TANGENT_CUTOFF: float = 6  # exponent of cutoff from 10**4ish to 10**7 m^-1
TANGENT_CUTOFF_TIME = -1  # same for time

# SI
do_scaling: bool = True
dx: float = 0.205 * (10 ** -6)  # meter per x pixel
dt: float = 1.0  # seconds per t pixel
boltzmann_constant: float = 1.36 * (10 ** -23) # Joules per Kelvin
room_temp: float = 298 # Kelvin