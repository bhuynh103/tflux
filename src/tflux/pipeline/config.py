# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:17:16 2025

@author: bhuyn
"""
from pathlib import Path
# Constants


DATA_DIR_PATH = Path('data/raw/temp/WT') # Swap all-data and temp

CSV_PATH = Path('data/processed_trimmed/LatB/experimental/metrics.csv')
CSV_PATHS = [Path('data/processed_trimmed/WT/metrics.csv'),
            Path('data/processed_trimmed/Bleb/control/metrics.csv'),
            Path('data/processed_trimmed/Bleb/experimental/metrics.csv'),
            Path('data/processed_trimmed/LatB/control/metrics.csv'),
            Path('data/processed_trimmed/LatB/experimental/metrics.csv')]

CROP_PERCENT: float = 0.3  # crops Grid half from left, half from right such that total cropping = CROP_PERCENT
WINDOW_SIZE: int = 7  # must be odd, if even adds 1
MAJORITY_PERCENT: float = 0.4
SUFFICIENT_COUNT: int = 1  # must be 1 ?
MIN_VALUE = 0  # floor of junction z-tilde ?

dx: float = 0.205 * (10 ** -6)  # meter per x pixel
dt: float = 1.0  # seconds per t pixel

TANGENT_CUTOFF: float = 6  # exponent of cutoff from 10**4ish to 10**7 m^-1
TANGENT_CUTOFF_TIME = -1  # same for time

boltzmann_constant = 1.36 * (10 ** -23) # Joules per Kelvin
room_temp = 298 # Kelvin

cmap1 = 'viridis'
cmap2 = 'spring'

# Settings
print_average_slopes = True
make_junc_summary = True
make_histogram = True
remake_histograms = False
