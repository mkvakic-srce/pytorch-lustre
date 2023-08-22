#!/bin/bash

#PBS -l ncpus=16
#PBS -l mem=200G

cd ${PBS_O_WORKDIR:-""}
module load scientific/pytorch
export NUM_DIRS=100
export NUM_FILES_PER_DIR=10000
export NUM_WORKERS=4
apptainer exec \
  --nv \
  --pwd /host_pwd \
  --bind ${PWD}:/host_pwd \
  --bind tempdata.sqsh:/tempdata:image-src=/ \
  $IMAGE_PATH python3 run_test_dataset_squashfs.py
