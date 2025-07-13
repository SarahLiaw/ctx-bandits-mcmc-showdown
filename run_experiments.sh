#!/bin/bash

# Script to run a sequence of experiments with error handling
# Usage: ./run_experiments.sh

set -e  # Exit on any error (but we'll handle it)

# Function to run a command and continue even if it fails
run_command() {
    echo "Running: $1"
    if eval "$1"; then
        echo "✓ Success: $1"
    else
        echo "✗ Failed: $1 (continuing...)"
    fi
    echo "----------------------------------------"
}

# Activate virtual environment
source venv/bin/activate

# Log start time
echo "Starting experiments at: $(date)"
echo "========================================"

# Run your sequence of commands
run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/adult-neuralTS.yaml --repeat 5 --log"

run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/magic-neuralTS.yaml --repeat 5 --log"


run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/covertype-neuralTS.yaml --repeat 5 --log"

run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/shuttle-neuralTS.yaml --repeat 5 --log"
# Add more commands as needed
# run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/other-dataset-neuralTS.yaml --repeat 5 --log"

# Log completion
echo "========================================"
echo "All experiments completed at: $(date)"
echo "Check individual command outputs above for any failures." 