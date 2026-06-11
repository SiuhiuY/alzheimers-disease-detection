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

# Print dataset information
DATASET_DIR=$(find . -maxdepth 1 -type d -name "ds*" | head -n 1)
NUM_RECORDINGS=$(find "$DATASET_DIR/derivatives" -name "*.set" | wc -l)
SIZE=$(du -sh "$DATASET_DIR" | cut -f1)
echo "[REPORT] Downloaded $NUM_RECORDINGS EEG recordings."
echo "[REPORT] Dataset size: $SIZE"
