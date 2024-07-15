#!/bin/bash

# Check if at least one YAML file path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_yaml_file1> <path_to_yaml_file2> ... <path_to_yaml_fileN>"
    exit 1
fi

# Loop through each YAML file path provided as arguments
for YAML_FILE_PATH in "$@"; do
    # Check if the file exists
    if [ ! -f "$YAML_FILE_PATH" ]; then
        echo "Error: File $YAML_FILE_PATH does not exist."
        continue
    fi

    # Extract the filename without extension and directory
    SESSION_NAME=$(basename "$YAML_FILE_PATH" .yaml)

    # Extract the GPU number from the filename (assuming it starts with gpuX)
    GPU_NUM=$(echo "$SESSION_NAME" | grep -oP '^gpu\K\d+')

    # Check if the GPU number was found
    if [ -z "$GPU_NUM" ]; then
        echo "Error: Unable to extract GPU number from filename."
        exit 1
    fi

    # Create or attach to the tmux session
    tmux new-session -d -s "$SESSION_NAME"

    # Send the command to the tmux session
    tmux send-keys -t "$SESSION_NAME" "conda activate augcon" C-m
    tmux send-keys -t "$SESSION_NAME" "CUDA_VISIBLE_DEVICES=${GPU_NUM} /home/hjyoon/projects/augcon/experiment.py --config ${YAML_FILE_PATH}" C-m

    echo "Started tmux session $SESSION_NAME with GPU $GPU_NUM for config $YAML_FILE_PATH."
done