#!/bin/bash

# Check if the config directory argument is provided
if [ $# -ne 3 ]; then
  echo "Usage: $0 <pretext> <seed> <gpu>"
  exit 1
fi

# Directory containing your YAML files
pretext="$1"
seed="$2"
gpu="$3"

# Loop through each .yaml file in the directory
for config_file in /mnt/sting/hjyoon/projects/aaa/configs/imwut/main_eval/*/"$pretext"/finetune/*/*/"$seed"/"$gpu"*.yaml; do
  # Check if the file exists
  if [ -e "$config_file" ]; then
    # Run your Python script with the config file as an argument serially
    ./experiment.py --config "$config_file"
  fi
done

wait