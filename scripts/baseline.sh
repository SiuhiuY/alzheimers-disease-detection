#!/bin/bash
#SBATCH -p GPU           
#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH --gres=gpu:1
#SBATCH -o /home/u961155/thesis_codes/jobs/%x.%j.out
#SBATCH -e /home/u961155/thesis_codes/jobs/%x.%j.err


# Load Conda
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate thesis-tf

# Move to correct directory
cd /home/u961155/thesis_codes/

# Debug: print location info
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

# Run Python
python -u eegnet_baseline.py