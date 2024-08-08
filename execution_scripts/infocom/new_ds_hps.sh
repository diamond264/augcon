#!/bin/bash

# Check if the config directory argument is provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <dir> <gpu>"
  exit 1
fi

# Directory containing your YAML files
dirname="$1"
gpu_id="$2"

# Loop through each .yaml file in the directory
for config_file in "$dirname"/*/*.yaml; do
  # Check if the file exists
  if [ -e "$config_file" ]; then
    # Run your Python script with the config file as an argument serially
    CUDA_VISIBLE_DEVICES="$gpu_id" python /home/hjyoon/projects/augcon/experiment.py --config "$config_file"
  fi
done