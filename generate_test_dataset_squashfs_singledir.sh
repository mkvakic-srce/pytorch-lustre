#!/bin/bash

module load scientific/pytorch
#run-command.sh python3 generate_test_dataset_squashfs_singledir.py
mksquashfs tempdata/single single.sqsh -processors 64
