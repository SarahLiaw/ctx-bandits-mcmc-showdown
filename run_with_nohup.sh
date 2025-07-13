#!/bin/bash

# Script to run experiments in background using nohup
# This will continue running even if you disconnect from SSH

# Create a log directory
mkdir -p logs

# Function to run a command with nohup
run_with_nohup() {
    local command="$1"
    local name="$2"
    local logfile="logs/${name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Starting $name in background..."
    echo "Log file: $logfile"
    
    nohup bash -c "
        cd $(pwd)
        source venv/bin/activate
        $command
    " > "$logfile" 2>&1 &
    
    echo "Process started with PID: $!"
    echo "----------------------------------------"
}

# Start time
echo "Starting experiments at: $(date)"
echo "========================================"

# Run your commands in parallel or sequence
run_with_nohup "python Neural/run_classifier.py --config_path Neural/configs/uci/adult-neuralTS.yaml --repeat 5 --log" "adult_neuralts"

run_with_nohup "python Neural/run_classifier.py --config_path Neural/configs/uci/covertype-neuralTS.yaml --repeat 5 --log" "covertype_neuralts"

run_with_nohup "python Neural/run_classifier.py --config_path Neural/configs/uci/magic-neuralTS.yaml --repeat 5 --log" "magic_neuralts"

# Wait for all background processes
echo "All commands started. Waiting for completion..."
wait

echo "========================================"
echo "All experiments completed at: $(date)"
echo "Check logs/ directory for individual log files." 