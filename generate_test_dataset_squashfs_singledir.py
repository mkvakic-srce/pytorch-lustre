#! /usr/bin/env python

import os
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np
from tqdm.auto import tqdm


DATASET_DIR = Path("tempdata/single")
NUM_DIRS = int(os.environ['NUM_DIRS'])
NUM_FILES_PER_DIR = int(os.environ['NUM_FILES_PER_DIR'])
ELEMENT_SIZE = int(os.environ['ELEMENT_SIZE'])

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == "__main__":
    create_dir(DATASET_DIR)

    def write_file(file):
        file_path = DATASET_DIR / f"FILE_{file:010}.npy"
        np.save(file_path, np.random.rand(ELEMENT_SIZE).astype(np.float32))

    Parallel(n_jobs=16)(delayed(write_file)(i) for i in range(NUM_DIRS*NUM_FILES_PER_DIR))

    for file in tqdm(range(NUM_DIRS*NUM_FILES_PER_DIR)):
        file_path = DATASET_DIR / f"FILE_{file:010}.npy"
        np.save(file_path, np.random.rand(ELEMENT_SIZE).astype(np.float32))
