#!/bin/bash

nseeds=8
firstseed=200

gpucount=-1
for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do

  gpucount=$(($gpucount + 1))
  gpu=$(($gpucount % 8))
  echo $seed $gpucount $gpu
  
  if [ "$1" = "lstm" ] || [ "$1" = "mclstm" ] || [ "$1" = "mcrlstm" ]; then
    model=$1
    note="ensemble"
    outfile="reports/global_${model}_$2.${seed}_${note}.out"

    if [ "$2" = "static" ]; then
      python3 main.py --gpu=$gpu --model_name=$1 --seed=$seed --no_static=False --concat_static=True train > $outfile &
    elif [ "$2" = "no_static" ]; then
      python3 main.py --gpu=$gpu --model_name=$1 --seed=$seed --no_static=True train > $outfile &
    else
      echo "bad model choice"
      exit
    fi

  else
    echo "bad model choice"
    exit
  fi

  #if [ $gpu -eq 7 ]; then
  #  wait
  #fi

done
