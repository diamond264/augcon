#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <output_file>"
  exit 1
fi

OUTPUT_FILE="$1"

# Execute the adb shell top command and store the output in the specified file
adb shell top -b -d 0.5 | grep --line-buffered "python" | tee "$OUTPUT_FILE"