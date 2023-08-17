#!/bin/bash

module load scientific/pytorch
export NUM_DIRS=100
export NUM_FILES_PER_DIR=10000
export NUM_WORKERS=4
run-command.sh python3 run_test_dataset.py
