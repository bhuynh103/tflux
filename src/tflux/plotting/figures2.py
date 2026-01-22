# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 09:12:38 2025

@author: bhuyn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import tflux.pipeline.config as config
import pandas as pd
import seaborn as sns
from scipy import stats

bleb_control = "/Users/bhuyn/Desktop/TFlux/figure-directories\WTvsBleb_control"
bleb_experimental = "/Users/bhuyn/Desktop/TFlux/figure-directories\WTvsBleb_experimental"
latb_control = "/Users/bhuyn/Desktop/TFlux/figure-directories\WTvsLabB_0.5uMLatB_control"
latb_experimental = "/Users/bhuyn/Desktop/TFlux/figure-directories\WTvsLabB_0.5uMLatB_experimental"

def fft_lineplot(df, treatment_type):
    
    control_df = df.loc[(df["treatment"] == f"{treatment_type}") & (df["group"] == "control")].sort_values(by=['junction', 'log_x'])
    experimental_df = df.loc[(df["treatment"] == f"{treatment_type}") & (df["group"] == "experimental")].sort_values(by=['junction', 'log_x'])
    
    # Plot best-fit line before noise floor at log(q) = 6
    noise_floor_mask = df['log_x'] < 6
    
    control_lr = linregress(control_df.loc[noise_floor_mask]['log_x'], control_df.loc[noise_floor_mask]['log_y'])
    control_lr_slope = control_lr.slope
    control_lr_intercept = control_lr.intercept
    control_lr_x = np.linspace(control_df['log_x'].min(), control_df['log_x'].max(), 100)
    control_lr_y = control_lr_slope * control_lr_x + control_lr_intercept
    
    experimental_lr = linregress(experimental_df.loc[noise_floor_mask]['log_x'], experimental_df.loc[noise_floor_mask]['log_y'])
    experimental_lr_slope = experimental_lr.slope
    experimental_lr_intercept = experimental_lr.intercept
    experimental_lr_x = np.linspace(experimental_df['log_x'].min(), experimental_df['log_x'].max(), 100)
    experimental_lr_y = experimental_lr_slope * experimental_lr_x + experimental_lr_intercept
    
    # Transform log data
    control_lr_x_power = 10 ** control_lr_x
    control_lr_y_power = 10 ** control_lr_y
    control_df['x'] = 10 ** control_df['log_x']
    control_df['y'] = 10 ** control_df['log_y']
    
    experimental_lr_x_power = 10 ** experimental_lr_x
    experimental_lr_y_power = 10 ** experimental_lr_y
    experimental_df['x'] = 10 ** experimental_df['log_x']
    experimental_df['y'] = 10 ** experimental_df['log_y']
    
    plt.plot(control_lr_x_power, control_lr_y_power, linewidth=3, color='blue')
    plt.plot(experimental_lr_x_power, experimental_lr_y_power, linewidth=3, color='red')
    
    # Plot all junctions
    max_junc = df['junction'].max()

    for junction_index in range(0, max_junc + 1):
        plt.plot(control_df.loc[control_df['junction'] == junction_index]['x'], 
                 control_df.loc[control_df['junction'] == junction_index]['y'],  
                 linestyle='-', 
                 color='blue',
                 linewidth=0.75,
                 alpha=0.4)
        
        plt.plot(experimental_df.loc[experimental_df['junction'] == junction_index]['x'], 
                 experimental_df.loc[experimental_df['junction'] == junction_index]['y'], 
                 linestyle='-', 
                 color='red',
                 linewidth=0.75,
                 alpha=0.4)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10**3.9, 10**6.5)
    plt.ylim(10**-24.6, 10**-17)
    plt.xlabel(r"Wavenumber ($m^{-1}$)")
    plt.ylabel(r"$\langle |u^2(q)| \rangle$ $(m^4)$")
    plt.grid()
    plt.legend()
    plt.show()
    
    treatment_df = df.loc[df['treatment'] == f'{treatment_type}']
    
    # fig, ax = plt.subplots
    # sns.lmplot(treatment_df, x='x', y='y')
    

def fft_scatterplot(df, treatment_type):
    
    control_df = df.loc[(df["treatment"] == f"{treatment_type}") & (df["group"] == "control")].sort_values(by=['junction', 'log_x'])
    experimental_df = df.loc[(df["treatment"] == f"{treatment_type}") & (df["group"] == "experimental")].sort_values(by=['junction', 'log_x'])
    
    # Plot best-fit line before noise floor at log(q) = 6
    noise_floor_mask = df['log_x'] < 6
    
    control_lr = linregress(control_df.loc[noise_floor_mask]['log_x'], control_df.loc[noise_floor_mask]['log_y'])
    control_lr_slope = control_lr.slope
    control_lr_intercept = control_lr.intercept
    control_lr_x = np.linspace(control_df['log_x'].min(), control_df['log_x'].max(), 100)
    control_lr_y = control_lr_slope * control_lr_x + control_lr_intercept
    
    experimental_lr = linregress(experimental_df.loc[noise_floor_mask]['log_x'], experimental_df.loc[noise_floor_mask]['log_y'])
    experimental_lr_slope = experimental_lr.slope
    experimental_lr_intercept = experimental_lr.intercept
    experimental_lr_x = np.linspace(experimental_df['log_x'].min(), experimental_df['log_x'].max(), 100)
    experimental_lr_y = experimental_lr_slope * experimental_lr_x + experimental_lr_intercept
    
    # Transform log data
    control_lr_x_power = 10 ** control_lr_x
    control_lr_y_power = 10 ** control_lr_y
    control_df['x'] = 10 ** control_df['log_x']
    control_df['y'] = 10 ** control_df['log_y']
    
    experimental_lr_x_power = 10 ** experimental_lr_x
    experimental_lr_y_power = 10 ** experimental_lr_y
    experimental_df['x'] = 10 ** experimental_df['log_x']
    experimental_df['y'] = 10 ** experimental_df['log_y']
    
    plt.plot(control_lr_x_power, control_lr_y_power, linewidth=3, color='blue')
    plt.plot(experimental_lr_x_power, experimental_lr_y_power, linewidth=3, color='red')
    
    # Plot all junctions
    max_junc = df['junction'].max()

    for junction_index in range(0, max_junc + 1):
        plt.scatter(control_df.loc[control_df['junction'] == junction_index]['x'], 
                 control_df.loc[control_df['junction'] == junction_index]['y'],   
                 color='blue',
                 s=0.5,
                 alpha=0.4)
        
        plt.scatter(experimental_df.loc[experimental_df['junction'] == junction_index]['x'], 
                 experimental_df.loc[experimental_df['junction'] == junction_index]['y'], 
                 linestyle='-', 
                 color='red',
                 s=0.5,
                 alpha=0.4)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10**3.9, 10**6.5)
    plt.ylim(10**-24.6, 10**-17)
    plt.xlabel(r"Wavenumber ($m^{-1}$)")
    plt.ylabel(r"$\langle |u^2(q)| \rangle$ $(m^4)$")
    plt.grid()
    plt.legend()
    plt.show()
    
    treatment_df = df.loc[df['treatment'] == f'{treatment_type}']
    
    # fig, ax = plt.subplots
    # sns.lmplot(treatment_df, x='x', y='y')


def fft_alphaplot(df, treatment_type):
    treatment_df = df.loc[df['treatment'] == f'{treatment_type}']

    # sns.displot(treatment_df, x='alpha', hue='group')
    sns.displot(treatment_df, x='alpha', hue='group', kde=True)
    # sns.displot(treatment_df, x='alpha', hue='group', element='step')
    # sns.displot(treatment_df, x='alpha', hue='group', element='step', kde=True)
    # sns.displot(treatment_df, x='alpha', hue='group', kind='kde', bw_adjust=1, fill=True)
    # sns.displot(treatment_df, x='alpha', hue='group', kind='kde', bw_adjust=0.5, fill=True)
    # sns.displot(treatment_df, x='alpha', hue='group', kind='kde', bw_adjust=0.25, fill=True)
    
def test_mean_alpha(df, treatment_type):
    if treatment_type == None:
        alpha_df = df.loc['alpha']
        t_control_vs_2, p_control_vs_2 = stats.ttest_1samp(alpha_df, 2.00)
        pass
    else:
        control_df = df.loc[(df["treatment"] == f"{treatment_type}") & (df["group"] == "control")]
        experimental_df = df.loc[(df["treatment"] == f"{treatment_type}") & (df["group"] == "experimental")]
    
        control_alpha = control_df["alpha"]
        experimental_alpha = experimental_df["alpha"]
        
        # print(control_alpha, experimental_alpha)
        
        t_control_vs_2, p_control_vs_2 = stats.ttest_1samp(control_alpha, 2.00)
        t_experimental_vs_2, p_experimental_vs_2 = stats.ttest_1samp(experimental_alpha, 2.00)
        t_control_vs_exp, p_control_vs_exp = stats.ttest_ind(control_alpha, experimental_alpha, equal_var=True)  # Welch’s t-test
        
        # Summarize results
        t_test_results = pd.DataFrame({
            f'{treatment_type} Comparison': ['Control vs 2.00', 'Experimental vs 2.00', 'Control vs Experimental'],
            't_stat': [t_control_vs_2, t_experimental_vs_2, t_control_vs_exp],
            'p_value': [p_control_vs_2, p_experimental_vs_2, p_control_vs_exp]
        })
    return t_test_results
