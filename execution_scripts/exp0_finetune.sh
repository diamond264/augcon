#!/bin/bash

# Define the list of configuration files and slice it based on the arguments
CONFIG_FILES=(/home/hjyoon/projects/augcon/config/exp0_hhar/finetune*/without_$1/*/*.yaml)

# Loop over the configuration files and execute the script with each one
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
  echo "Running with config file: $CONFIG_FILE"
  python experiment.py --config "$CONFIG_FILE" && echo "Finished running with config file: $CONFIG_FILE"
done