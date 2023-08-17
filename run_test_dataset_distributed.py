#! /usr/bin/env python

from pathlib import Path

import time
import numpy as np
import os
import torch
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader, DataLoaderShard
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

DATASET_DIR = Path("tempdata")
NUM_DIRS = int(os.environ['NUM_DIRS'])
NUM_FILES_PER_DIR = int(os.environ['NUM_FILES_PER_DIR'])
BATCH_SIZE = 256

class TestDataset(Dataset):

    def __init__(self, root_path, num_dirs, num_files_per_dir, batch_size, accelerator):
        super().__init__()

        self.root_path = root_path
        self.num_dirs = num_dirs
        self.num_files_per_dir = num_files_per_dir
        self.num_processes = accelerator.num_processes
        self.process_index = accelerator.process_index
        self.length = num_dirs * num_files_per_dir * self.num_processes
        self.batch_size = batch_size
        self.data_path = os.path.join(DATASET_DIR,
                                      f'distributed_{self.process_index:02}.npy')
        self.data = np.load(self.data_path)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_process = idx%self.batch_size + (idx//self.num_processes//self.batch_size)*self.batch_size
        return torch.from_numpy(self.data[idx_process])

def main():

    accelerator = Accelerator()

    dataset = TestDataset(DATASET_DIR,
                          NUM_DIRS,
                          NUM_FILES_PER_DIR,
                          BATCH_SIZE,
                          accelerator)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )
    dataloader = accelerator.prepare(dataloader)

    progress_bar = tqdm(
        range(len(dataloader)),
        disable=not accelerator.is_local_main_process,
        leave=True,
        dynamic_ncols=True,
    )
    s = 0
    for elem in dataloader:
        s += float(torch.mean(elem))
        progress_bar.update()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        files = len(dataset)
        elapsed = progress_bar.format_dict['elapsed']
        file_rate = files/elapsed
        lines = ['s: %0.2f' % s,
                 'files: %d' % files,
                 'elapsed: %0.2f' % elapsed,
                 'file_rate: %0.2f' % file_rate]
        print(','.join(lines))

if __name__ == "__main__":
    main()
