#!/bin/bash

module load scientific/pytorch
export NUM_DIRS=100
export NUM_FILES_PER_DIR=100000
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
ngpus=$(egrep -o '[0-9]' <<< $CUDA_VISIBLE_DEVICES | wc -l)
apptainer run \
  --nv \
  --pwd /host_pwd \
  --bind ${PWD}:/host_pwd \
  $IMAGE_PATH \
    accelerate launch \
      --multi_gpu \
      --gpu_ids $CUDA_VISIBLE_DEVICES \
      --num_processes $ngpus \
      run_test_dataset_distributed.py
