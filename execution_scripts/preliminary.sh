#!/bin/bash

# set the directory where the config files are located
configs_dir="/home/hjyoon/projects/augcon/config/finetune/preliminary"

# set the log directory
log_dir="/mnt/sting/hjyoon/projects/augcontrast/logs"

# set the number of GPUs available
num_gpus=8
num_parallel=24

# set the python file to execute
run_py="/home/hjyoon/projects/augcon/run.py"

# initialize counter variable for number of scripts executed
script_counter=0

# function to cancel all child processes
function cancel_processes {
  echo "Cancelling all child processes..."
  pkill -P $$
}

# trap SIGTERM signal and cancel child processes
trap 'cancel_processes' SIGTERM

# get the PID of the script and echo it
pid=$$
echo "Script PID: $pid"

# loop through all config files in the directory
for config_file in "$configs_dir"/*.yaml; do
  # check if we've already executed 8 scripts
  if (( $script_counter % $num_parallel == 0 )); then
    # if so, wait for all processes to finish before executing the next batch
    wait
  fi

  # determine the GPU index to use for this task
  gpu_index=$(( $script_counter % $num_gpus ))
  
  # set the CUDA_VISIBLE_DEVICES environment variable
  export CUDA_VISIBLE_DEVICES="$gpu_index"
  
  # extract the base filename (without extension) from the config file path
  base_filename=$(basename "$config_file" .yaml)

  cmd="python $run_py --config $config_file --finetune &> $log_dir/${base_filename}.log &"
  echo "Executing command: $cmd"
  
  # execute the python script with the appropriate config file and log the output
  python "$run_py" --config "$config_file" --finetune &> "$log_dir/${base_filename}.log" &
  
  # increment the script counter
  (( script_counter++ ))
done

# wait for all processes to finish
wait
