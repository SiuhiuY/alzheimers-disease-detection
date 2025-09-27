#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-36:00
#SBATCH --gres=gpu:1
#SBATCH -o ../results/%x.%j.out
#SBATCH -e ../results/%x.%j.err

source ~/.bashrc
conda activate thesis

cd $HOME/thesis_codes/src
python eeg_processor.py