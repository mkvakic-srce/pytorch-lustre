#! /usr/bin/env python

import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


DATASET_DIR = Path("tempdata")
NUM_DIRS = int(os.environ['NUM_DIRS'])
NUM_FILES_PER_DIR = int(os.environ['NUM_FILES_PER_DIR'])
ELEMENT_SIZE = int(os.environ['ELEMENT_SIZE'])
NSHARDS = int(os.environ['NSHARDS'])

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == "__main__":
    create_dir(DATASET_DIR)
    for shard in range(NSHARDS):
        file_path = os.path.join(DATASET_DIR,
                                 f"distributed_{shard:02}.npy")
        np.save(file_path,
                np.random.rand(NUM_DIRS*NUM_FILES_PER_DIR, ELEMENT_SIZE).astype(np.float32))
