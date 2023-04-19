#!/bin/bash

# Parse command-line arguments
START_INDEX=$1
END_INDEX=$2

# Define the list of configuration files and slice it based on the arguments
CONFIG_FILES=(/home/hjyoon/projects/augcon/config/pretrain/preliminary/*.yaml)
CONFIG_FILES=("${CONFIG_FILES[@]:START_INDEX:END_INDEX-START_INDEX}")

# Loop over the configuration files and execute the script with each one
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
  echo "Running with config file: $CONFIG_FILE"
  python run.py --config "$CONFIG_FILE" --pretrain && echo "Finished running with config file: $CONFIG_FILE"
done