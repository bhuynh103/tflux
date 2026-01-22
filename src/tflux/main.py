# -*- coding: utf-8 -*-
### TODO: omega scaling distribution

"""
Created on Wed Dec 11 00:33:25 2024

@author: Brain Huynh
"""

from tflux.pipeline import config, run


def main():
    sample = None
    if config.process_sample_directory:
        sample = run.run_pipeline(data_dir_path=config.DATA_DIR_PATH)   # Save slope data to metrics.csv
    
    return sample # N surfaces

# Main Execution 
if __name__ == "__main__":
    sample = main()


'''
Pair processing code block, TODO: find a home

batch_plot_df = None
alpha_df = None
t_tests = None

if config.process_batch_directory:
    batch_plot_df, alpha_df = pair_analyze_samples(config.PAIR_DIRECTORY)
    
    t_tests = []
    bleb_t_test = figures2.test_mean_alpha(alpha_df, "Bleb")
    latb_t_test = figures2.test_mean_alpha(alpha_df, "LatB")
    t_tests = [bleb_t_test, latb_t_test]
    bleb_t_test.to_csv('t_test_results.csv', mode='w')
    latb_t_test.to_csv('t_test_results.csv', mode='a')
'''