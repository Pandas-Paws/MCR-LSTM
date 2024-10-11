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

# Initialize the ensemble dictionary
ens_dict = {}

# This loop will run the evaluation procedure for all ensembles
for seed in range(firstSeed, firstSeed + nSeeds):

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
    
    # Grab the test output file for this split
    file_seed = run_dir.split('seed')[1]
    results_file = glob.glob(f"{run_dir}/*lstm*seed{file_seed}.p")[0]
    
    with open(results_file, 'rb') as f:
        seed_dict = pickle.load(f)

    # Define a list of all possible variables
    possible_vars = ['qsim', 'trashcell', 'cellstate', 'mrflow', 'outflow']

    # Create the ensemble dictionary
    for basin in seed_dict:
        if seed == firstSeed:
            # Filter existing columns to rename
            existing_cols = {var: f"{var}_{seed}" for var in possible_vars if var in seed_dict[basin].columns}
            ens_dict[basin] = seed_dict[basin].rename(columns=existing_cols)
        else:
            # Filter existing columns to include in merging
            existing_cols = [var for var in possible_vars if var in seed_dict[basin].columns]
            merged_data = seed_dict[basin][existing_cols].rename(columns={var: f"{var}_{seed}" for var in existing_cols})
            ens_dict[basin] = pd.merge(
                ens_dict[basin], merged_data,
                how='inner',
                left_index=True,
                right_index=True
            )

# Calculate ensemble mean for variables that exist
for basin in ens_dict:
    for variable in possible_vars:
        if any(variable in col for col in ens_dict[basin].columns):
            simdf = ens_dict[basin].filter(regex=f'{variable}_')
            if not simdf.empty:
                ens_mean = simdf.mean(axis=1)
                ens_dict[basin].insert(0, variable, ens_mean)

# Save the ensemble results as a pickle
fname = f"analysis/results_data/internal_state_{experiment}_{note}.pkl"
with open(fname, 'wb') as f:
    pickle.dump(ens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
