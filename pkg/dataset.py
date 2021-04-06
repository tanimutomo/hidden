from dataclasses import dataclass
import os
import io
import typing

from PIL import Image
import torch

import pkg.wordvec


Image = torch.FloatTensor
Message = typing.Union[torch.FloatTensor, pkg.wordvec.WordVector]


@dataclass
class DataItem:
    img Image
    msg Message

    def img(self) -> Image:
        return self.img

    def msg(self) -> torch.FloatTensor:
        if self.is_msg_tensor():
            return self.msg
        else:
            return self.msg.vec

    def is_msg_tensor(self) -> bool:
        return true if isinstance(self.msg, torch.FloatTensor) else false


class ByteMessageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, msg_len, img_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            msg_len (int): Length for byte message.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.msg_len = int(msg_len)
        self.img_transform = img_transform
        self.files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return sum(os.path.isfile(os.path.join(self.root_dir, name)) for name in os.listdir(self.root_dir))

    def __getitem__(self, idx: int) -> DataItem:
        img = Image.open(os.path.join(self.root_dir, self.files[idx]))
        msg = torch.rand(self.msg_len).round()
        if self.img_transform:
            img = self.img_transform(img)
        return img, msg


class WordMessageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_words, word_vecs, img_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            num_words (int): Number of words are embedded to image.
            word_vecs (tensor): Word embedding vector.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.num_words = int(num_words)
        self.word_vec = word_vec
        self.img_transform = img_transform
        self.files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return sum(os.path.isfile(os.path.join(self.root_dir, name)) for name in os.listdir(self.root_dir))

    def __getitem__(self, idx: int) -> DataItem:
        img = Image.open(os.path.join(self.root_dir, self.files[idx]))
        wvec = self.word_vec.get_with_random(self.num_words)
        if self.img_transform:
            img = self.img_transform(img)
        return img, wvec


@dataclass
class DataStat:
    mean: float
    std: float


@dataclass
class DatasetStats:
    y: DataStat
    u: DataStat
    v: DataStat


@dataclass
class COCODatasetStats:
    # use the mean and the std of https://en.wikipedia.org/wiki/YUV
    y: DataStat =DataStat(mean=0.5, std=0.500)
    u: DataStat =DataStat(mean=0.0, std=0.436)
    v: DataStat =DataStat(mean=0.0, std=0.615)

