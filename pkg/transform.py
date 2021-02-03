from dataclasses import dataclass
import typing

import kornia
import torch
from torchvision import (
    transforms,
)

@dataclass
class ImageTransformer(object):
    img_size: typing.Tuple[int, int] =(128, 128)

    def __post_init__(self):
        self.train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.img_size, scale=(0.3, 1.0)),
            transforms.ToTensor(),
            kornia.color.RgbToYuv(),
        ])

        self.test = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            kornia.color.RgbToYuv(),
        ])

        self.post_process = transforms.Compose([
            kornia.color.YuvToRgb(),
        ])
