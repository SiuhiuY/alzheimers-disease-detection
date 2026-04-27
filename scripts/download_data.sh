#!/usr/bin/env bash 

# Safety flags 
set -euo pipefail

# Print status message to user 
echo "[INFO] Preparing data directory structure..." 

# Create data directory 
mkdir -p data
cd data

# Download data 
echo "[INFO] Downloading..." 
bash ../scripts/openneuro_download.sh
echo "[INFO] Data downloaded"
