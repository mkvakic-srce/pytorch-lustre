#! /usr/bin/env python

import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


DATASET_DIR = Path("tempdata")
NUM_DIRS = int(os.environ['NUM_DIRS'])
NUM_FILES_PER_DIR = int(os.environ['NUM_FILES_PER_DIR'])
ELEMENT_SIZE = int(os.environ['ELEMENT_SIZE'])

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == "__main__":
    create_dir(DATASET_DIR)
    np.save(os.path.join(DATASET_DIR, 'memory.npy'),
            np.random.rand(NUM_DIRS*NUM_FILES_PER_DIR, ELEMENT_SIZE).astype(np.float32))
