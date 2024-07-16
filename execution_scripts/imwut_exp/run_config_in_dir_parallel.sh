#!/bin/bash

# Check if the config directory argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <config_dir>"
  exit 1
fi

# Directory containing your YAML files
config_dir="$1"

# Check if the specified directory exists
if [ ! -d "$config_dir" ]; then
  echo "Error: Directory '$config_dir' does not exist."
  exit 1
fi

# Loop through each .yaml file in the directory
for config_file in "$config_dir"/*.yaml; do
  # Check if the file exists
  if [ -e "$config_file" ]; then
    # Run your Python script with the config file as an argument in the background
    SESSION_NAME=$(basename "$config_file" .yaml)
    GPU_NUM=$(echo "$SESSION_NAME" | grep -oP '^gpu\K\d+')
    CUDA_VISIBLE_DEVICES=${GPU_NUM} ./experiment.py --config "$config_file" &
  fi
done

# Wait for all background processes to finish
wait