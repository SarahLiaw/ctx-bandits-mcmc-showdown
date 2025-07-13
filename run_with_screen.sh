#!/bin/bash

# Script to run experiments using screen sessions
# Each command runs in its own screen session

# Function to create a screen session and run a command
run_in_screen() {
    local session_name="$1"
    local command="$2"
    
    echo "Creating screen session: $session_name"
    echo "Command: $command"
    
    # Create a new screen session and run the command
    screen -dmS "$session_name" bash -c "
        cd $(pwd)
        source venv/bin/activate
        echo 'Starting $session_name at $(date)'
        $command
        echo 'Finished $session_name at $(date)'
        sleep 10
    "
    
    echo "Screen session '$session_name' created"
    echo "To attach: screen -r $session_name"
    echo "To list sessions: screen -ls"
    echo "----------------------------------------"
}

# Start time
echo "Starting experiments with screen sessions at: $(date)"
echo "========================================"

# Run your commands in separate screen sessions
run_in_screen "adult_neuralts" "python Neural/run_classifier.py --config_path Neural/configs/uci/adult-neuralTS.yaml --repeat 5 --log"

run_in_screen "covertype_neuralts" "python Neural/run_classifier.py --config_path Neural/configs/uci/covertype-neuralTS.yaml --repeat 5 --log"

run_in_screen "magic_neuralts" "python Neural/run_classifier.py --config_path Neural/configs/uci/magic-neuralTS.yaml --repeat 5 --log"

echo "========================================"
echo "All screen sessions created!"
echo "Useful commands:"
echo "  screen -ls                    # List all sessions"
echo "  screen -r session_name        # Attach to a session"
echo "  screen -S session_name -X quit # Kill a session"
echo "  Ctrl+A, D                     # Detach from current session" 