#!/bin/bash

#PBS -l ngpus=2
#PBS -l ncpus=16
#PBS -l mem=200G

cd ${PBS_O_WORKDIR:-""}
module load scientific/pytorch
export NUM_DIRS=100
export NUM_FILES_PER_DIR=10000
device_ids=""
for device in $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); do
  device_id=$(nvidia-smi -L | grep $device | egrep -o 'GPU [0-3]' | sed 's/GPU //g')
  device_ids+=$device_id,
done
export CUDA_VISIBLE_DEVICES=${device_ids//,$/}
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
