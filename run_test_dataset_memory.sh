#!/bin/bash

module load scientific/pytorch
export NUM_DIRS=10
export NUM_FILES_PER_DIR=1000
export NUM_WORKERS=4
run-command.sh python3 run_test_dataset_memory.py