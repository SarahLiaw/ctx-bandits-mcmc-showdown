#!/bin/bash

# Usage: ./run_experiments.sh

set -e  

run_command() {
    echo "Running: $1"
    if eval "$1"; then
        echo "Success: $1"
    else
        echo "Failed: $1 (continuing...)"
    fi
    echo "----------------------------------------"
}

# source venv/bin/activate

echo "Starting experiments at: $(date)"
echo "========================================"

run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/financial-neuralTS.yaml --repeat 5 --log"
run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/financial-fgneuralts.yaml --repeat 5 --log"
run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/financial-sfgneuralts.yaml --repeat 5 --log"

run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/jester-neuralTS.yaml --repeat 5 --log"
run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/jester-fgneuralts.yaml --repeat 5 --log"
run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/jester-sfgneuralts.yaml --repeat 5 --log"

# run_command "python Neural/run_classifier.py --config_path Neural/configs/uci/other-dataset-neuralTS.yaml --repeat 5 --log"

echo "========================================"
echo "All experiments completed at: $(date)"
echo "Check individual command outputs above for any failures." 