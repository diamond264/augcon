#!/bin/bash

# Parse command-line arguments
START_INDEX=$1
END_INDEX=$2

# Define the list of configuration files and slice it based on the arguments
CONFIG_FILES=(/home/hjyoon/projects/augcon/config/exp0_hhar/pretrain/*/*_perdomain_8421.yaml)
echo "Total number of config files: ${#CONFIG_FILES[@]}"

# Loop over the configuration files and execute the script with each one
for ((i=START_INDEX; i<=END_INDEX && i<${#CONFIG_FILES[@]}; i++)); do
  CONFIG_FILE="${CONFIG_FILES[$i]}"
  echo "Running with config file: $CONFIG_FILE"
  python experiment.py --config "$CONFIG_FILE" && echo "Finished running with config file: $CONFIG_FILE"
done