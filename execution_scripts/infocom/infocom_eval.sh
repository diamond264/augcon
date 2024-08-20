#!/bin/bash

# Check if the root directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <root_dir>"
  exit 1
fi

# Set the root directory
ROOT_DIR=$1

# Iterate over all {shot_num}/{setting}/{seed} directories
for shot_dir in "$ROOT_DIR"/*/; do
  for setting_dir in "$shot_dir"*/; do
    for seed_dir in "$setting_dir"*/; do
      
      echo "Processing directory: $seed_dir"

      # Get all YAML files in the current {shot_num}/{setting}/{seed} directory
      yaml_files=("$seed_dir"*.yaml)
      
      # Execute all YAML files in parallel
      for yaml_file in "${yaml_files[@]}"; do
        echo "Executing: $yaml_file"
        
        # Replace `your_command_to_run_yaml_files` with your actual command
        # your_command_to_run_yaml_files "$yaml_file" &
        SESSION_NAME=$(basename "$yaml_file" .yaml)
        GPU_NUM=$(echo "$SESSION_NAME" | grep -oP '^gpu\K\d+')
        CUDA_VISIBLE_DEVICES=${GPU_NUM} ./experiment.py --config "$yaml_file" &
      done
      
      # Wait for all parallel tasks in the current directory to complete
      wait

      echo "Finished processing directory: $seed_dir"

    done
  done
done

echo "All directories have been processed."
