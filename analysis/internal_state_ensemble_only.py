"""
This file is part of the accompanying code to our manuscript:
Y. Wang, L. Zhang, N.B. Erichson, T. Yang. (2025). A Mass Conservation Relaxed (MCR) LSTM Model for Streamflow Simulation
"""

import pickle
import sys

import numpy as np
import pandas as pd

from performance_functions import (baseflow_index, bias, flow_duration_curve,
                                   get_quant, high_flows, low_flows, nse, alpha_nse, beta_nse,
                                   kge, stdev_rat, zero_freq, FHV, FLV, mass_balance)


# file name of ensemble dictionary is a user input
experiment = sys.argv[1]
note = sys.argv[2]

# load ensemble file
fname = f"results_data/internal_state_{experiment}_{note}.pkl"
with open(fname, 'rb') as f:
    ens_dict = pickle.load(f)

# calcualte performance measures for ensembles
stats = []
bdex = -1
for basin in ens_dict:

    # list columns in dataframe
    sim_cols = ens_dict[basin].filter(regex="qsim")
    _, nMembers = ens_dict[basin].filter(regex="qsim").shape

    # calcualte ensemble mean performance metrics
    obs5, sim5 = get_quant(ens_dict[basin], 0.05)
    obs95, sim95 = get_quant(ens_dict[basin], 0.95)
    obs0, sim0 = zero_freq(ens_dict[basin])
    obsH, simH = high_flows(ens_dict[basin])
    obsL, simL = low_flows(ens_dict[basin])
    e_fhv = FHV(ens_dict[basin]) # yhwang 20240616
    e_flv = FLV(ens_dict[basin]) # yhwang 20240616
    e_nse = nse(ens_dict[basin])
    e_nse_alpha = alpha_nse(ens_dict[basin]) # yhwang 20240616
    e_nse_beta = beta_nse(ens_dict[basin]) # yhwang 20240616
    e_kge, r, alpha, beta = kge(ens_dict[basin]) # yhwang 20240616
    massbias_total, massbias_pos, massbias_neg = mass_balance(ens_dict[basin]) # yhwang 20240617
    e_bias = bias(ens_dict[basin])
    e_stdev_rat = stdev_rat(ens_dict[basin])
    #  obsBF, simBF = baseflow_index(ens_dict[basin])
    obsFDC, simFDC = flow_duration_curve(ens_dict[basin])

    # add ensemble mean stats to globaldictionary
    stats.append({
        'basin': basin,
        'nse': e_nse,
        'alpha_nse': e_nse_alpha,
        'beta_nse': e_nse_beta, 
        'kge': e_kge,
        'kge_r': r,
        'kge_alpha': alpha,
        'kge_beta': beta,  
        'fhv': e_fhv,
        'flv': e_flv, 
        'massbias_total': massbias_total,
        'massbias_pos': massbias_pos,
        'massbias_neg': massbias_neg, 
        'bias': e_bias,
        'stdev': e_stdev_rat,
        'obs5': obs5,
        'sim5': sim5,
        'obs95': obs95,
        'sim95': sim95,
        'obs0': obs0,
        'sim0': sim0,
        'obsL': obsL,
        'simL': simL,
        'obsH': obsH,
        'simH': simH,
        'obsFDC': obsFDC,
        'simFDC': simFDC
    })

    # print basin-specific stats
    bdex = bdex + 1
    print(f"{basin} ({bdex} of {len(ens_dict)}) --- NSE: {stats[bdex]['nse']:.3f}, KGE: {stats[bdex]['kge']:.3f}")

# save ensemble stats as a csv file
stats_df = pd.DataFrame(stats,
                     columns=[
                         'basin', 'nse', 'alpha_nse', 'beta_nse', 'kge', 'kge_r', 'kge_alpha', 'kge_beta',
                         'fhv', 'flv', 'massbias_total', 'massbias_pos', 'massbias_neg',
                         'bias', 'stdev', 'obs5', 'sim5', 'obs95', 'sim95', 'obs0',
                         'sim0', 'obsL', 'simL', 'obsH', 'simH', 'obsFDC', 'simFDC'
                     ])

# Calculate mean and median, excluding NaN values
mean_stats = stats_df.mean(skipna=True)
median_stats = stats_df.median(skipna=True)

# Add mean and median rows to the DataFrame
mean_stats['basin'] = 'mean'
median_stats['basin'] = 'median'
stats_df = stats_df.append(mean_stats, ignore_index=True)
stats_df = stats_df.append(median_stats, ignore_index=True)

# Save ensemble stats as a CSV file
fname = f"stats/{experiment}_{note}.csv"
stats_df.to_csv(fname, index=False)

# Print to screen
print(f"Mean NSE: {mean_stats['nse']:.3f}, Mean KGE: {mean_stats['kge']:.3f}")
print(f"Median NSE: {median_stats['nse']:.3f}, Median KGE: {median_stats['kge']:.3f}")
print(f"Num Failures: {np.sum((stats_df['nse'] < 0).values.ravel())}")
