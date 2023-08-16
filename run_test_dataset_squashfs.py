#! /usr/bin/env python

from pathlib import Path

import time
import numpy as np
import os
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


DATASET_DIR = Path("/tempdata")
NUM_DIRS = int(os.environ['NUM_DIRS'])
NUM_FILES_PER_DIR = int(os.environ['NUM_FILES_PER_DIR'])
NUM_WORKERS = int(os.environ['NUM_WORKERS'])
BATCH_SIZE = 64

class TestDataset(Dataset):

    def __init__(self, root_path, num_dirs, num_files_per_dir):
        super().__init__()

        self.root_path = root_path
        self.num_dirs = num_dirs
        self.num_files_per_dir = num_files_per_dir
        self.length = num_dirs * num_files_per_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dir = idx // self.num_files_per_dir
        file = idx % self.num_files_per_dir
        data_item_path = os.path.join(
            self.root_path, f"{dir:010}", f"FILE_{file:010}.npy"
        )
        return torch.from_numpy(np.load(data_item_path))

if __name__ == "__main__":

    accelerator = Accelerator()

    dataset = TestDataset(DATASET_DIR, NUM_DIRS, NUM_FILES_PER_DIR)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    dataloader = accelerator.prepare(dataloader)

    s = 0
    progress_bar = tqdm(
        range(len(dataloader)),
        disable=not accelerator.is_local_main_process,
        leave=True,
        dynamic_ncols=True,
    )
    for elem in dataloader:
        s += float(torch.mean(elem))
        progress_bar.update()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        files = NUM_DIRS*NUM_FILES_PER_DIR
        elapsed = progress_bar.format_dict['elapsed']
        file_rate = files/elapsed
        lines = ['s: %0.2f' % s,
                 'files: %d' % files,
                 'elapsed: %0.2f' % elapsed,
                 'file_rate: %0.2f' % file_rate]
        print(','.join(lines))
