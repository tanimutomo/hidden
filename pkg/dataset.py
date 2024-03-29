from dataclasses import dataclass
import os
import io
import typing

from PIL import Image
import torch


class ByteMessageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, msg_len, img_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.msg_len = int(msg_len)
        self.img_transform = img_transform
        self.files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return sum(os.path.isfile(os.path.join(self.root_dir, name)) for name in os.listdir(self.root_dir))

    def __getitem__(self, idx: int):
        img = Image.open(os.path.join(self.root_dir, self.files[idx]))
        msg = torch.rand(self.msg_len).round()
        if self.img_transform:
            img = self.img_transform(img)
        return img, msg


@dataclass
class DataStat:
    mean: float
    std: float


@dataclass
class DatasetStats:
    y: DataStat
    u: DataStat
    v: DataStat

    def means(self) -> typing.List[float]:
        return [self.y.mean, self.u.mean, self.v.mean]

    def stds(self) -> typing.List[float]:
        return [self.y.std, self.u.std, self.v.std]


@dataclass
class COCODatasetStats:
    # use the mean and the std of https://en.wikipedia.org/wiki/YUV
    y: DataStat =DataStat(mean=0.5, std=0.500)
    u: DataStat =DataStat(mean=0.0, std=0.436)
    v: DataStat =DataStat(mean=0.0, std=0.615)

    def means(self) -> typing.List[float]:
        return [self.y.mean, self.u.mean, self.v.mean]

    def stds(self) -> typing.List[float]:
        return [self.y.std, self.u.std, self.v.std]
