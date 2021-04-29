from dataclasses import dataclass
import os
import sys
import typing

from PIL import Image
import torch
import torchvision

sys.path.append(os.path.abspath("."))

import pkg.transform
import pkg.dataset


NUM_WORKERS = 4
PIN_MEMORY = True


@dataclass
class DataController:
    resol: int
    dataset_stats: pkg.dataset.DatasetStats
    train_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    train_batch_size: int =0
    test_batch_size: int =0
    require_trainset: bool =True
    require_testset: bool =True

    def __post_init__(self):
        img_transformer = pkg.transform.ImageTransformer(self.resol, self.dataset_stats)

        if self.require_trainset:
            self.train_dataset.img_transform = img_transformer.train
            self.train_loader =  torch.utils.data.DataLoader(
                self.train_dataset, self.train_batch_size, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=pkg.dataset.BatchItem,
            )

        if self.require_testset:
            self.test_dataset.img_transform = img_transformer.test
            self.test_loader =  torch.utils.data.DataLoader(
                self.test_dataset, self.test_batch_size, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=pkg.dataset.BatchItem,
            )

        self.img_post_transformer = img_transformer.post_process if img_transformer.post_process else None

