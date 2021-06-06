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
            ToRGB(),
            transforms.ToTensor(),
            kornia.color.RgbToYuv(),
            transforms.Normalize(self.dataset_stats.means(), self.dataset_stats.stds()),
        ])

        self.test = transforms.Compose([
            ToRGB(),
            transforms.ToTensor(),
            kornia.color.RgbToYuv(),
            transforms.Normalize(self.dataset_stats.means(), self.dataset_stats.stds()),
        ])

        self.save = transforms.Compose([
            Unnormalize(self.dataset_stats.means(), self.dataset_stats.stds()),
            kornia.color.YuvToRgb(),
            Clamp(0.0, 1.0),
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
        return x * std + mean

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Clamp:
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: torch.FloatTensor):
        return torch.clamp(x, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '()'
