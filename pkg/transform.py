from dataclasses import dataclass
import typing

import kornia
from PIL import Image
import torch
from torchvision import transforms

import pkg.dataset


@dataclass
class ImageTransformer(object):
    img_size: typing.Tuple[int, int]
    dataset_stats: pkg.dataset.DatasetStats

    def __post_init__(self):
        self.train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.img_size, scale=(0.3, 1.0)),
            ToRGB(),
            transforms.ToTensor(),
            kornia.color.RgbToYuv(),
            transforms.Normalize(self.dataset_stats.means(), self.dataset_stats.stds()),
        ])

        self.test = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            ToRGB(),
            transforms.ToTensor(),
            kornia.color.RgbToYuv(),
            transforms.Normalize(self.dataset_stats.means(), self.dataset_stats.stds()),
        ])

        self.save = transforms.Compose([
            Unnormalize(self.dataset_stats.means(), self.dataset_stats.stds()),
            kornia.color.YuvToRgb(),
        ])

        self.psnr = Unnormalize(self.dataset_stats.means(), self.dataset_stats.stds())


class ToRGB:
    def __call__(self, img: Image.Image):
        return img.convert("RGB")

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Unnormalize:
    def __init__(self, mean: typing.List[float], std: typing.List[float]):
        self.mean = mean
        self.std = std
    
    def __call__(self, x: torch.FloatTensor):
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        if mean.ndim == 1:
            mean = mean[:, None, None]
        if std.ndim == 1:
            std = std[:, None, None]
        x.mul_(std).add_(mean)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

