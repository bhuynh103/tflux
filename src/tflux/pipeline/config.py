# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:17:16 2025

@author: bhuyn
"""
# Constants
SINGLE_DIRECTORY = "/Users/bhuyn/Desktop/TFlux/WT-selected"
PAIR_DIRECTORY = "/Users/bhuyn/Desktop/TFlux/figure-directories-paired"

CROP_PERCENT = 5
WINDOW_SIZE = 7
MAJORITY_PERCENT = 0.4
SUFFICIENT_COUNT = 1
MIN_VALUE = 0

dx = 0.205 * (10 ** -6)  # meter per x pixel
dt = 1.0  # seconds per t pixel

TANGENT_CUTOFF = 6
TANGENT_CUTOFF_TIME = -1

boltzmann_constant = 1.36 * (10 ** -23) # Joules per Kelvin
room_temp = 298 # Kelvin

cmap1 = 'viridis'
cmap2 = 'spring'

# Settings
process_sample_directory = True
find_average_metrics = True
include_visualizations = True

process_batch_directory = False
include_figures = True