#!/bin/bash

# Define the log file name with a timestamp
LOG_FILE="main_$(date +%Y%m%d_%H%M%S).log"

# Run the Python script, display output in the terminal, and log it to the file
python3 main.py | tee "$LOG_FILE"

# Print a message to indicate where the log file is saved
echo "Script execution completed. Logs saved to $LOG_FILE"