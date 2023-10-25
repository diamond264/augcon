import os
import argparse
import pandas as pd
from glob import glob

def parse(input, output=None):
    df = pd.DataFrame(columns=['cpu', 'mem'])
    
    with open(input, 'r') as f:
        lines = f.readlines()
    
    pid = -1
    accumulated_cpu = 0
    accumulated_mem = 0
    for e, line in enumerate(lines):
        if e == 0: pid = line.strip().split()[0]
        l = line.strip()
        if len(l) > 0:
            vals = l.split()
            cpu = float(vals[8].strip())
            mem = float(vals[9].strip())
            accumulated_cpu += cpu
            accumulated_mem += mem
            
        if l.split()[0] == pid:
            new_df = pd.DataFrame({'cpu': [accumulated_cpu], 'mem': [accumulated_mem]})
            df = pd.concat([df, new_df], ignore_index=True)
            accumulated_cpu = 0
            accumulated_mem = 0
    
    # resample from 2Hz to 1Hz (by averaging)
    # average every two rows
    df = df.groupby(df.index // 2).mean()
    
    if output is not None:
        df.to_csv(output, index=False)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse adb rsc output')
    parser.add_argument('-i', '--input', help='Input file')
    parser.add_argument('-o', '--output', help='Output file')
    args = parser.parse_args()
    parse(args.input, args.output)