# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:17:16 2025

@author: bhuyn
"""
from pathlib import Path
from datetime import datetime

# Pipeline Settings
data_dir_path: Path = Path("data/raw/all-data/WT")
save_pickle: bool = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = f"logs/tflux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
make_pdfs = True
save_average_slope_csv = True
make_junc_summary = True
include_bad_junctions_in_summary = False
make_histogram = False
remake_histograms = False
save_cells = True
save_config = True


# k-means hyperparams
K_MEANS: int = 4
K_MEANS_ITER: int = 40
SMOOTH_ITER: int = 3    # Lower for better performace
LAMBDA: float = 0.9
MIN_ISLAND_FACES: int = 10000
SEED: int = 0
NORMAL_WEIGHT: float = 1.0
GEOM_WEIGHT: float = 0.5
PERCENT_ZERO_THRESHOLD: float = 1.00

# grid interpolation hyperparams
CROP_PERCENT: float = 0.3  # crops (CROP_PERCENT / 2) from left and right
WINDOW_SIZE: int = 7
MAJORITY_PERCENT: float = 0.4
SUFFICIENT_COUNT: int = 1
MIN_VALUE = 0

# Noise floor threshold
TANGENT_CUTOFF: float = 6  # exponent of cutoff from 10**4ish to 10**7 m^-1
TANGENT_CUTOFF_TIME = -1  # same for time

CSV_PATH = Path('data/processed_trimmed/LatB/experimental/metrics.csv')
CSV_PATHS = [Path('data/processed_trimmed/WT/metrics.csv'),
            Path('data/processed_trimmed/Bleb/control/metrics.csv'),
            Path('data/processed_trimmed/Bleb/experimental/metrics.csv'),
            Path('data/processed_trimmed/LatB/control/metrics.csv'),
            Path('data/processed_trimmed/LatB/experimental/metrics.csv')]

# SI
do_scaling: bool = True
dx: float = 0.205 * (10 ** -6)  # meter per x pixel
dt: float = 1.0  # seconds per t pixel
boltzmann_constant: float = 1.36 * (10 ** -23) # Joules per Kelvin
room_temp: float = 298 # Kelvin

# plotting
cmap1 = 'viridis'
cmap2 = 'spring'