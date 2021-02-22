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


@dataclass
class DataController:
    train_dataset_path: str
    test_dataset_path: str
    train_batch_size: int
    test_batch_size: int
    msg_len: int
    resol: int

    def __post_init__(self):
        img_transformer = pkg.transform.ImageTransformer(self.resol)

        train_dataset = pkg.dataset.WatermarkDataset(self.train_dataset_path, self.msg_len, img_transformer.train)
        test_dataset = pkg.dataset.WatermarkDataset(self.test_dataset_path, self.msg_len, img_transformer.test)

        self.train_loader =  torch.utils.data.DataLoader(train_dataset, self.train_batch_size, True)
        self.test_loader =  torch.utils.data.DataLoader(test_dataset, self.test_batch_size, False)

        self.img_post_transformer = img_transformer.post_process if img_transformer.post_process else None

