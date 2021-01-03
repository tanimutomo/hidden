from dataclasses import dataclass
import typing

import kornia
import torch
from torchvision import (
    transforms,
)

@dataclass
class Transformer(object):
    img_size: typing.Tuple[int, int] =(128, 128)

    def train_image_transform(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.img_size, scale=(0.3, 1.0)),
            transforms.ToTensor(),
            kornia.color.RgbToYuv(),
        ])

    def test_image_transform(self):
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            kornia.color.RgbToYuv(),
        ])

    def model_output_transform(self):
        return transforms.Compose([
            kornia.color.YuvToRgb(),
        ])
