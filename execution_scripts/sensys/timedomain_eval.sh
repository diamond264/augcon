#!/bin/bash

# Check if the root directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <root_dir>"
  exit 1
fi

# Set the root directory
ROOT_DIR=$1

echo "Processing directory: $ROOT_DIR"

# Get all YAML files in the current {shot_num}/{setting}/{seed} directory
yaml_files=("$ROOT_DIR"/*.yaml)

# Execute all YAML files in parallel
for yaml_file in "${yaml_files[@]}"; do
  echo "Executing: $yaml_file"
  
  # Replace `your_command_to_run_yaml_files` with your actual command
  # your_command_to_run_yaml_files "$yaml_file" &
  SESSION_NAME=$(basename "$yaml_file" .yaml)
  GPU_NUM=$(echo "$SESSION_NAME" | grep -oP '^gpu\K\d+')
  echo "CUDA_VISIBLE_DEVICES=$GPU_NUM ./experiment.py --config $yaml_file &"
  CUDA_VISIBLE_DEVICES=${GPU_NUM} ./experiment.py --config "$yaml_file" &
done

echo "All directories have been processed."
