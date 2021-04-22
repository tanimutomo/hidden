import os
import sys
import typing

import distortion
import torch
import torch.nn as nn

sys.path.append(os.path.abspath("."))

import pkg.architecture


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_parallel: bool = False
        self.module_names: typing.List[str] = []

    def parallel(self, device_ids: typing.List[int]):
        if len(device_ids) <= 1: return
        for m in self.module_names:
            setattr(self, m, torch.nn.DataParallel(getattr(self, m), device_ids=device_ids))
        self._is_parallel = True

    def state_dict(self):
        if not self._is_parallel:
            return super().state_dict()
        sd = dict()
        for m in self.module_names:
            for k, v in getattr(self, m).module.state_dict().items():
                sd[f"{m}.{k}"] = v
        return sd


class HiddenModel(_Base):
    def __init__(
        self,
        train_distorter: distortion.Distorter,
        test_distorter: distortion.Distorter,
        train_distortion_parallelable: bool,
        test_distortion_parallelable: bool,
    ):
        super().__init__()
        self.encoder = pkg.architecture.Encoder()
        self.train_distorter = train_distorter
        self.test_distorter = test_distorter
        self.decoder = pkg.architecture.Decoder()
        self.module_names = ["encoder", "decoder"]
        if train_distortion_parallelable: self.module_names.append("train_distorter")
        if test_distortion_parallelable: self.module_names.append("test_distorter")

    def forward(self, img: torch.FloatTensor, msg: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
        enc_img = self.encoder(img, msg)
        dis = self.train_distorter if self.training else self.test_distorter
        dis_img = dis(img, enc_img)
        pred_msg = self.decoder(dis_img)
        return enc_img, dis_img, pred_msg


class Discriminator(_Base):
    def __init__(self):
        super().__init__()
        self.discriminator = pkg.architecture.Discriminator()
        self.module_names = ["discriminator"]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.discriminator(x)
