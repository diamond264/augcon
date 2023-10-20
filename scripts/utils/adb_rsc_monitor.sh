#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <PID> <output_file>"
  exit 1
fi

PID="$1"
OUTPUT_FILE="$2"

# Execute the adb shell top command and store the output in the specified file
adb shell top -b -d 1 -p "$PID" | tee "$OUTPUT_FILE" | grep "$PID"