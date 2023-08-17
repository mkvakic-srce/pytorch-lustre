#!/bin/bash

cd ${PBS_O_WORKDIR}
module load scientific/pytorch
export NUM_DIRS=100
export NUM_FILES_PER_DIR=10000
export ELEMENT_SIZE=2048
run-command.sh python3 generate_test_dataset_squashfs_singledir.py
mksquashfs tempdata/single single.sqsh 2>/dev/null
