# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:17:16 2025

@author: bhuyn
"""
# Constants
DIRECTORY = "/Users/bhuyn/Desktop/TFlux/WTvsLabB_0.5uMLatB_experimental"
WINDOW_SIZE = 7
MAJORITY_PERCENT = 0.1
SUFFICIENT_COUNT = 1

dx = 0.205 * (10 ** -6)  # meter per x step
dt = 1.0  # seconds per t step

TANGENT_CUTOFF = 6

boltzmann_constant = 1.36 * (10 ** -23) # Joules per Kelvin
room_temp = 298 # Kelvin

# Settings
include_visualizations = True
find_average_metrics = True

### DIRECTORY PATHS ###
# "/Users/bhuyn/Desktop/TFlux/WT-selected"
# "/Users/bhuyn/Desktop/TFlux/WTvsLabB_0.5uMLatB_experimental"