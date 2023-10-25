#!/bin/bash

# Check if the config directory argument is provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <pretext> <prefix>"
  exit 1
fi

# Directory containing your YAML files
pretext="$1"
prefix="$2"

# Loop through each .yaml file in the directory
for config_file in /mnt/sting/hjyoon/projects/aaa/configs/imwut/main_hps/*/"$pretext"/finetune/*/"$prefix"*.yaml; do
  # Check if the file exists
  if [ -e "$config_file" ]; then
    # Run your Python script with the config file as an argument serially
    ./experiment.py --config "$config_file"
  fi
done