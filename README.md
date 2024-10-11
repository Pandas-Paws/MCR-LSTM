# MCR-LSTM
Codes for A Mass Conservation Relaxed (MCR) LSTM Model for Streamflow Simulation with a Large-Scale Verification Experiment over 531 Watersheds across CONUS

The codes are adopted, modified based on, and organized very similarly to https://github.com/kratzert/lstm_for_pub. 

Thus, this instruction will be in a similar fashion as in https://github.com/kratzert/lstm_for_pub.

# To start with:
Get CAMELS data from https://ral.ucar.edu/solutions/products/camels. The filepath must be: './data/basin_dataset_public_v1p2' and must include the CAMELS attributes as a subdirectory: './data/basin_dataset_public_v1p2/camels_attributes_v2.0'.

Download the updated NLDAS forcings from HydroShare. These include daily min and max temperature, compared to the CAMELS NLDAS forcings that only contain daily mean temperature.

# To train models:
Run the training scripts: 'train_global.sh'. Options for the global script are: (i) the model type: 'lstm', 'mclstm', 'mcrlstm', and (ii) the option to use catchment attributes as static input features: 'static' or 'no_static'. You can also specify a note for your model. For example, in the provided train_global.sh, the note is specified as "ensemble".

Examples: ./train_global.sh lstm static => This will train an LSTM model with 32 inputs (i.e., static attributes included as input variables)
         ./train_global.sh mcrlstm static => This will train a MCR-LSTM model with 32 inputs.
         ./train_global.sh mclstm no_static => This will train a MC-LSTM model with only 5 hydrometeorological inputs. None of the models used in this study uses "no_static".

The trained models will be saved under a folder "runs/". The log file will be saved under a folder "reports/" with name "global_${model}_$2.${seed}_${note}.out". 

# To test models:
Run the test scripts: 'run_global_parallel.py'. Options for these include (i) the experiment name (i.e., global_${model}_${static/no_static}), (ii) the note you specified for your trained models, and (iii) the epoch you want to test.  

Example: python run_global.py global_mcrlstm_static ensemble 28 => This will test the LSTM trained with 32 inputs (i.e., with static variables) after being trained after completing 28 epochs training. The trained LSTM model has a note "ensemble". 

Outputs from the test runs are stored in CSV (human-readable) files in the './analysis/resutls/' subdirectory. 

# To evaluate the models:
In the 'analysis' subdirectory, run the 'main_performance_ensemble_only.py' script to get ensemble performance statistics. Options include: (i) the expeirment name (i.e., global_${model}_${static/no_static}, (ii) the note you specified for your trained models, and (iii) the epoch you want to test.
These statistics are stored in the './analysis/stats' subdirectory.

Example: python analysis/main_performance_ensemble_only.py global_mcrlstm_static ensemble 28 => This will provide the statistical performance of MCR-LSTM trained with 32 inputs after completing 28 epochs training. The trained LSTM model has a note "ensemble". 

