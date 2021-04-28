from dataclasses import dataclass
import os
import io
import typing

import PIL
import torch

import pkg.wordvec


Image = torch.FloatTensor
Message = typing.Union[torch.FloatTensor, pkg.wordvec.WordVector]


@dataclass
class BatchItem:
    img: torch.Tensor
    msg: typing.Union[torch.Tensor, typing.List[pkg.wordvec.WordVector]]

    def msg_vec(self) -> torch.FloatTensor:
        return self.msg if self.is_msg_tensor() else self.msg.vec

    def is_msg_tensor(self) -> bool:
        return True if isinstance(self.msg, torch.FloatTensor) else False


class _Base(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_transform):
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return sum(os.path.isfile(os.path.join(self.root_dir, name)) for name in os.listdir(self.root_dir))

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        img = PIL.Image.open(os.path.join(self.root_dir, self.files[idx]))
        if self.img_transform: img = self.img_transform(img)
        msg = self._get_messages()
        return img, msg

    def _get_messages(self):
        raise NotImplementedError()


class BitMessageDataset(_Base):
    def __init__(self, root_dir, msg_len, img_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            msg_len (int): Length for byte message.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, img_transform=img_transform)
        self.msg_len = int(msg_len)

    def _get_messages(self) -> torch.Tensor:
        return torch.rand(self.msg_len).round()


class WordMessageDataset(_Base):
    def __init__(self, root_dir, num_words, word_vec, img_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            num_words (int): Number of words are embedded to image.
            word_vec (tensor): Word embedding vector.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, img_transform=img_transform)
        self.num_words = int(num_words)
        self.word_vec = word_vec

    def _get_messages(self) -> torch.Tensor:
        return self.word_vec.get_with_random(self.num_words)


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
