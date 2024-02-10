from joblib import Parallel, delayed
import subprocess
import shlex
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Run ./patch.sh in parallel with joblib')

# Add an argument for the input file
parser.add_argument('input_file', help='Path to the input file')

# Parse the command-line arguments
args = parser.parse_args()

# Read the file line by line
with open(args.input_file, 'r') as file:
    lines = file.read().splitlines()

# Rest of the code remains the same
# Function to run the script
def run_script(arg):
    subprocess.run(shlex.split(f'./patch.sh {args.input_file}'))

# Run the script in parallel with joblib
Parallel(n_jobs=128)(delayed(run_script)(arg) for arg in lines)