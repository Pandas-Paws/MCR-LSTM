"""
This file is part of the accompanying code to our manuscript:
Y. Wang, L. Zhang, N.B. Erichson, T. Yang. (2025). A Mass Conservation Relaxed (MCR) LSTM Model for Streamflow Simulation
"""

import glob
import os
import pickle
import sys
import pandas as pd
from multiprocessing import Pool

# number of ensemble members
#seeds = [202, 210, 208, 201, 211, 204, 200, 215]
nSeeds = 1 # based on how many ensemble members you run
firstSeed = 200
seeds = [200, 201, 202, 203, 204, 205, 206, 207]

# number of GPUs available
nGPUs = 8

# user inputs
experiment = sys.argv[1]
note = sys.argv[2]
epoch_num = sys.argv[3]

def run_evaluation(seed_gpu):
    seed, gpu = seed_gpu
    
    # get the correct run directory by reading the screen report
    fname = f"reports/{experiment}.{seed}_{note}.out"
    print(f"Working on seed: {seed} -- file: {fname}")
    
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        if "Sucessfully stored basin attributes in" in line:
            full_path = line.split('attributes in ')[1].strip()  # Extract the full path
            run_dir = os.path.dirname(full_path)
            
            
    print("==== run_dir: ", run_dir)

    run_command = f"python3 main.py --gpu={gpu} --run_dir={run_dir} --epoch_num={epoch_num} evaluate"
    os.system(run_command)
    
    
    # grab the test output file for this split
    file_seed = run_dir.split('seed')[1]
    results_file = glob.glob(f"{run_dir}/*lstm*seed{file_seed}_epoch{epoch_num}.p")[0]
    #results_file = glob.glob(f"{run_dir}/*mcrlstm*seed{file_seed}_epoch{epoch_num}.p")[0]
    print('results_file: ', results_file)
    
    with open(results_file, 'rb') as f:
        seed_dict = pickle.load(f)

    # create the ensemble dictionary
    for basin in seed_dict:
        seed_dict[basin].rename(columns={'qsim': f"qsim_{seed}"}, inplace=True)
    return seed_dict, seed

def merge_dicts(dict1, dict2, seed):
    for basin in dict2:
        if basin in dict1:
            dict1[basin] = pd.merge(
                dict1[basin],
                dict2[basin][f"qsim_{seed}"],
                how='inner',
                left_index=True,
                right_index=True)
        else:
            dict1[basin] = dict2[basin]
    return dict1

if __name__ == "__main__":
    # Determine which GPUs to use for each seed
    seeds_gpus = [(seed, seed % nGPUs) for seed in seeds] #range(firstSeed, firstSeed + nSeeds)
    
    # Use a Pool with a number of processes equal to the number of GPUs
    with Pool(processes=nGPUs) as pool:
        results = pool.map(run_evaluation, seeds_gpus)

    # Combine the results into a single dictionary
    ens_dict, _ = results[0]
    for result in results[1:]:
        seed_dict, seed = result
        ens_dict = merge_dicts(ens_dict, seed_dict, seed)

    # calculate ensemble mean
    for basin in ens_dict:
        simdf = ens_dict[basin].filter(regex='qsim_')
        ensMean = simdf.mean(axis=1)
        ens_dict[basin].insert(0, 'qsim', ensMean)

    # save the ensemble results as a pickle
    fname = f"analysis/results_data/{experiment}_{note}_{epoch_num}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(ens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
