#!/bin/bash

nseeds=1
firstseed=200

# Check if GPU argument is provided
if [ -z "$3" ]; then
  echo "Usage: ./train.sh <model> <static/no_static> <gpu_id>"
  exit 1
fi

gpu=$3

for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do
  echo "Running seed: $seed on GPU: $gpu"
  
  if [ "$1" = "lstm" ] || [ "$1" = "mclstm" ] || [ "$1" = "mcrlstm" ]; then
    model=$1
    note="seed200"
    outfile="reports/global_${model}_$2.${seed}_${note}.out"

    if [ "$2" = "static" ]; then
      python3 main.py --gpu=$gpu --model_name=$1 --seed=$seed --no_static=False --concat_static=True train > $outfile &
    elif [ "$2" = "no_static" ]; then
      python3 main.py --gpu=$gpu --model_name=$1 --seed=$seed --no_static=True train > $outfile &
    else
      echo "Bad model choice"
      exit 1
    fi

  else
    echo "Bad model choice"
    exit 1
  fi

  # Wait for all background processes to complete before starting a new one
  wait
done
