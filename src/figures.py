# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 07:31:32 2025

@author: bhuyn
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

import config


def violin_comparison():
    Bleb_control = []
    Bleb_experimental = []
    LatB_control = []
    LatB_experimental = []
    
    df = pd.DataFrame({
        "Tension (N/m)": Bleb_control + Bleb_experimental + LatB_control + LatB_experimental,
        "Group": ["Control"] * len(Bleb_control) + ["Experimental"] * len(Bleb_experimental) + ["Control"] * len(LatB_control) + ["Experimental"] * len(LatB_experimental),
        "Drug": ["Bleb"] * (len(Bleb_control) + len(Bleb_experimental)) + ["LatB"] * (len(LatB_control) + len(LatB_experimental))
    })
    sns.violinplot(df, x="Tension (N/m)", y="Drug", hue="Group", log_scale=True)
    return


def violin_tension(tension_list):
    sns.violinplot(tension_list, orient="h", log_scale=True, color="skyblue", edgecolor="black")
    median_val = np.median(tension_list)
    plt.axvline(median_val, color="red", linestyle="--", linewidth=2, label=f"Median = {median_val:.2e}")
    mean_val = np.mean(tension_list)
    plt.axvline(mean_val, color="blue", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2e}")
    plt.legend()
    plt.xlabel(r"Effective Tension $(N/m)$")
    plt.title("Tension Distribution")
    
    plt.show()


def hist_slope(tension_list, bins):
    plt.hist(
        tension_list,
        bins=bins,
        edgecolor="black"   # bar borders
    )
    
    plt.xlabel(r"$\beta$")   # LaTeX-style alpha
    plt.ylabel("Count")
    plt.show()
    
    
def fft_scatterplot(df):
    x = df["log_x"]
    y = df["log_y"]
    slope, intercept, _, _, _ = linregress(x, y)
    
    # Map categories to colors automatically
    samples = df['sample'].unique()
    colors = plt.cm.tab10(range(len(samples)))  # pick distinct colors
    color_map = dict(zip(samples, colors))
    
    # Plot each category
    plt.figure(figsize=(6, 4))
    for samp in samples:
        subset = df[df['sample'] == samp]
        junctions = subset['junction'].unique()
        for junction in junctions:
            subsubset = subset[subset['junction'] == junction]
            plt.plot(subsubset['log_x'], subsubset['log_y'], color=color_map[samp], linewidth=0.5, alpha=0.5)
        
        slope, intercept, _, _, _ = linregress(subset['log_x'], subset['log_y'])
        x_fit = np.linspace(subset['log_x'].min(), subset['log_x'].max(), 1000)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color=color_map[samp], linewidth=2)
    plt.show()