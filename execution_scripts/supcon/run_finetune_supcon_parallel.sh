#!/bin/bash

# Loop through each .yaml file in the directory
for seed in 0 1 2 3 4; do
  for config_file in /mnt/sting/hjyoon/projects/aaa/configs/supervised_adaptation/main_eval/sup/ichar/metasimclr/finetune/10shot/linear/seed"$seed"/*.yaml; do
    # Check if the file exists
    if [ -e "$config_file" ]; then
      CUDA_VISIBLE_DEVICES="$seed" ./experiment.py --config "$config_file" &
    fi
  done
done

wait