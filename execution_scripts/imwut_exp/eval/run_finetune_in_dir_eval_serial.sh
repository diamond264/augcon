#!/bin/bash

# Check if the config directory argument is provided
if [ $# -ne 4 ]; then
  echo "Usage: $0 <pretext> <setting> <seed> <gpu>"
  exit 1
fi

# Directory containing your YAML files
pretext="$1"
setting="$2"
seed="$3"
gpu="$4"

# Loop through each .yaml file in the directory
for config_file in /mnt/sting/hjyoon/projects/aaa/configs/imwut/main_eval/*/"$pretext"/finetune/*/"$setting"/"$seed"/"$gpu"*.yaml; do
  # Check if the file exists
  if [ -e "$config_file" ]; then
    # Run your Python script with the config file as an argument serially
    ./experiment.py --config "$config_file"
  fi
done

wait