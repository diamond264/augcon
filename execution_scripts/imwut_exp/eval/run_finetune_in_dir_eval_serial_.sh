#!/bin/bash

# Check if the config directory argument is provided
if [ $# -ne 3 ]; then
  echo "Usage: $0 <pretext> <setting> <seed> <gpu>"
  exit 1
fi

# Directory containing your YAML files
pretext="$1"
setting="$2"
seed="$3"

# Loop through each .yaml file in the directory
for i in {0..7}; do
  ./execution_scripts/imwut_exp/eval/run_finetune_in_dir_eval_serial.sh "$pretext" "$setting" "$seed" gpu$i &
done

wait