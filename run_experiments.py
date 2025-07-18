#!/usr/bin/env python3

import subprocess
import time
import logging
from datetime import datetime
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_command(command, timeout=3600):
    logging.info(f"Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            logging.info(f"✓ Success: {command}")
            if result.stdout:
                logging.info(f"Output: {result.stdout}")
            return True
        else:
            logging.error(f"✗ Failed: {command}")
            logging.error(f"Return code: {result.returncode}")
            if result.stderr:
                logging.error(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logging.error(f"✗ Timeout: {command} (after {timeout}s)")
        return False
    except Exception as e:
        logging.error(f"✗ Exception: {command} - {str(e)}")
        return False

def main():
    logging.info("=" * 50)
    logging.info("Starting experiment sequence")
    logging.info(f"Start time: {datetime.now()}")
    logging.info("=" * 50)
    commands = [
        "source venv/bin/activate && python Neural/run_classifier.py --config_path Neural/configs/uci/adult-neuralTS.yaml --repeat 5 --log",
        "source venv/bin/activate && python Neural/run_classifier.py --config_path Neural/configs/uci/covertype-neuralTS.yaml --repeat 5 --log",
        "source venv/bin/activate && python Neural/run_classifier.py --config_path Neural/configs/uci/magic-neuralTS.yaml --repeat 5 --log",
    ]
    results = []
    for i, command in enumerate(commands, 1):
        logging.info(f"\n--- Command {i}/{len(commands)} ---")
        success = run_command(command)
        results.append((command, success))
        time.sleep(5)
    logging.info("\n" + "=" * 50)
    logging.info("EXPERIMENT SUMMARY")
    logging.info("=" * 50)
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    logging.info(f"Total commands: {len(results)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    if failed > 0:
        logging.info("\nFailed commands:")
        for command, success in results:
            if not success:
                logging.info(f"  - {command}")
    logging.info(f"\nEnd time: {datetime.now()}")
    logging.info("=" * 50)

if __name__ == "__main__":
    main() 