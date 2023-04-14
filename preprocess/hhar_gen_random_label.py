import pandas as pd
import random
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Add randomDomain column to CSV file')
parser.add_argument('input', type=str, help='Input CSV file name')
parser.add_argument('output', type=str, help='Output CSV file name')
parser.add_argument('interval', type=int, help='Interval for randomDomain assignment')
args = parser.parse_args()

# Read the input CSV file into a Pandas DataFrame
df = pd.read_csv(args.input)

# Initialize the randomDomain column with -1 for all rows
df["randomDomain"] = -1

# Generate randomDomain values for every n rows (specified by interval argument)
for i in range(0, len(df), args.interval):
    random_domain = random.randint(0, 9)
    df.loc[i:i+args.interval-1, "randomDomain"] = random_domain

# Save the updated DataFrame to the output CSV file
df.to_csv(args.output, index=False)