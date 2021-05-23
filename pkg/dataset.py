from dataclasses import dataclass
import os
import io
import typing

import PIL
import torch

import pkg.wordvec


Image = torch.Tensor
Message = typing.Union[torch.Tensor, pkg.wordvec.Pair]

@dataclass
class Item:
    img: torch.Tensor
    msg: typing.Union[torch.Tensor, pkg.wordvec.Pair]
    _w2v: pkg.wordvec.GloVe =None

    def msg_vec(self) -> torch.Tensor:
        return self.msg if self.msg_is_tensor() else self.msg.vec

    def msg_is_tensor(self) -> bool:
        return True if isinstance(self.msg, torch.Tensor) else False


class BatchItem:
    _img: torch.Tensor
    _msg: typing.Union[torch.Tensor, typing.List[pkg.wordvec.WordVector]]

    def __init__(self, items: typing.Tuple[Item]):
        self._img = torch.stack([item.img for item in list(items)], dim=0)
        if items[0].msg_is_tensor():
            self._msg = torch.stack([item.msg for item in list(items)], dim=0)
        else:
            self._msg = pkg.wordvec.WordVector(
                idx=torch.stack([torch.tensor(item.msg.idx) for item in list(items)], dim=0),
                vec=torch.stack([item.msg.vec for item in list(items)], dim=0),
            )

    def img(self):
        return self._img

    def msg(self):
        return self._msg

    def msg_vec(self) -> torch.Tensor:
        return self._msg if self.msg_is_tensor() else self._msg.serialized()

    def msg_is_tensor(self) -> bool:
        return True if isinstance(self._msg, torch.Tensor) else False

    def to_(self, device: torch.device):
        self._img = self._img.to(device)
        self._msg = self._msg.to(device)


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
        return self._construct_item(img, msg)

    def _get_messages(self):
        raise NotImplementedError()

    def _construct_item(self, img, msg) -> Item:
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

    def _construct_item(self, img, msg) -> Item:
        return Item(img=img, msg=msg)


class WordMessageDataset(_Base):
    def __init__(self, root_dir, word_vec, img_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            word_vec (tensor): Word embedding vector.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, img_transform=img_transform)
        self.wvec = word_vec

    def _get_messages(self) -> pkg.wordvec.Pair:
        return self.wvec.get_with_random()

    def _construct_item(self, img, msg) -> Item:
        return Item(img=img, msg=msg, _w2v=self.wvec)


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
    y: DataStat =DataStat(mean=0.452, std=0.253)
    u: DataStat =DataStat(mean=-0.022, std=0.058)
    v: DataStat =DataStat(mean=0.022, std=0.079)

    def means(self) -> typing.List[float]:
        return [self.y.mean, self.u.mean, self.v.mean]

    def stds(self) -> typing.List[float]:
        return [self.y.std, self.u.std, self.v.std]
