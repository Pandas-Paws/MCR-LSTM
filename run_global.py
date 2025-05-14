import glob
import os
import pickle
import sys

import pandas as pd

# number of ensemble members
nSeeds = 1
firstSeed = 200

# user inputs
experiment = sys.argv[1]
note = sys.argv[2]
gpu = sys.argv[3]

# This loop will run the evaluation procedure for all ensembles
for seed in range(firstSeed, firstSeed + nSeeds):  # loop through randomized ensemble
    #for seed in range(210,211):

    # get the correct run directory by reading the screen report
    if note == "":
        fname = f"reports/{experiment}.{seed}.out"
    else:
        fname = f"reports/{experiment}.{seed}_{note}.out"
    print(f"Working on seed: {seed} -- file: {fname}")
    f = open(fname)
    with open(fname, 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if "Sucessfully stored basin attributes in" in line:
            full_path = line.split('attributes in ')[1].strip()  # Extract the full path
            run_dir = os.path.dirname(full_path)
    print("==== run_dir: ", run_dir)
    
    run_command = f"python3 main.py --gpu={gpu} --run_dir={run_dir} evaluate"
    os.system(run_command)
    
    # grab the test output file for this split
    file_seed = run_dir.split('seed')[1]
    results_file = glob.glob(f"{run_dir}/*lstm*seed{file_seed}.p")[0]
    with open(results_file, 'rb') as f:
        seed_dict = pickle.load(f)

    # create the ensemble dictionary
    for basin in seed_dict:
        seed_dict[basin].rename(columns={'qsim': f"qsim_{seed}"}, inplace=True)
    if seed == 200:
        ens_dict = seed_dict
    else:
        for basin in seed_dict:
            ens_dict[basin] = pd.merge(
                ens_dict[basin],
                seed_dict[basin][f"qsim_{seed}"],
                how='inner',
                left_index=True,
                right_index=True)

# --- end of seed loop -----------------------------------------

# calculate ensemble mean
for basin in ens_dict:
    simdf = ens_dict[basin].filter(regex='qsim_')
    ensMean = simdf.mean(axis=1)
    ens_dict[basin].insert(0, 'qsim', ensMean)

# save the ensemble results as a pickle
fname = f"analysis/results_data/{experiment}_{note}.pkl"
with open(fname, 'wb') as f:
    pickle.dump(ens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
