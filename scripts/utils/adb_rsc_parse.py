import os
import argparse
import pandas as pd
from glob import glob

def parse(pid, input, output=None):
    df = pd.DataFrame(columns=['cpu', 'mem'])
    
    with open(input, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        l = line.strip()
        if l.startswith(str(pid)):
            vals = l.split()
            cpu = float(vals[8].strip())
            mem = float(vals[9].strip())
            new_df = pd.DataFrame({'cpu': [cpu], 'mem': [mem]})
            df = pd.concat([df, new_df], ignore_index=True)
    
    if output is not None:
        df.to_csv(output, index=False)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse adb rsc output')
    parser.add_argument('-i', '--input', help='Input file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-p', '--pid', help='PID')
    args = parser.parse_args()
    parse(args.pid, args.input, args.output)