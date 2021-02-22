from dataclasses import dataclass
import typing

import kornia
from PIL import Image
import torch
from torchvision import transforms


@dataclass
class ImageTransformer(object):
    img_size: typing.Tuple[int, int] =(128, 128)

    def __post_init__(self):
        self.train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.img_size, scale=(0.3, 1.0)),
            ToRGB(),
            transforms.ToTensor(),
            kornia.color.RgbToYuv(),
        ])

        self.test = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            ToRGB(),
            transforms.ToTensor(),
            kornia.color.RgbToYuv(),
        ])

        self.post_process = transforms.Compose([
            kornia.color.YuvToRgb(),
        ])


class ToRGB:
    def __call__(self, img: Image.Image):
        return img.convert("RGB")

    def __repr__(self):
        return self.__class__.__name__ + '()'
