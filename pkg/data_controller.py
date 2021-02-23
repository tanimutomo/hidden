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
    msg_len: int
    resol: int
    train_dataset_path: str =""
    test_dataset_path: str =""
    train_batch_size: int =0
    test_batch_size: int =0
    require_trainset: bool =True
    require_testset: bool =True

    def __post_init__(self):
        img_transformer = pkg.transform.ImageTransformer(self.resol)

        if self.require_trainset:
            train_dataset = pkg.dataset.WatermarkDataset(self.train_dataset_path, self.msg_len, img_transformer.train)
            self.train_loader =  torch.utils.data.DataLoader(train_dataset, self.train_batch_size, True)

        if self.require_testset:
            test_dataset = pkg.dataset.WatermarkDataset(self.test_dataset_path, self.msg_len, img_transformer.test)
            self.test_loader =  torch.utils.data.DataLoader(test_dataset, self.test_batch_size, False)

        self.img_post_transformer = img_transformer.post_process if img_transformer.post_process else None

