#!/bin/bash

# Check if device_name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <device_name>"
  exit 1
fi

# Assign the argument to a variable
DEVICE_NAME=$1

# Create output directory if it doesn't exist
mkdir -p overhead_output

echo "Running experiment on ${DEVICE_NAME}..."
echo "python experiment.py --config overhead_configs/config_metacpc.yaml >> overhead_output/${DEVICE_NAME}_metacpc_overhead.txt"
# Execute the command and redirect output to the file
python experiment.py --config overhead_configs/config_metacpc.yaml >> "overhead_output/${DEVICE_NAME}_metacpc_overhead.txt"
echo "Experiment completed. Output saved to overhead_output/${DEVICE_NAME}_metacpc_overhead.txt"