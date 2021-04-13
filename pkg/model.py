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

    def _parallel(self, device_ids: typing.List[int]):
        if len(device_ids) <= 1: return
        for m in self.module_names:
            setattr(self, m, torch.nn.DataParallel(getattr(self, m), device_ids=device_ids))
        self._is_parallel = True

    def state_dict(self):
        if not self._is_parallel:
            return self.state_dict()
        sd = dict()
        for m in self.module_names:
            for k, v in getattr(self, m).module.state_dict().items():
                sd[f"{m}.{k}"] = v
        return sd


class HiddenModel(_Base):
    def __init__(self, distortioner: distortion.Distortioner, distortion_parallel: bool):
        super().__init__()
        self.encoder = pkg.architecture.Encoder()
        self.distortioner = distortioner
        self.decoder = pkg.architecture.Decoder()
        self.module_names = ["encoder", "decoder"]
        if distortion_parallel:
            self.module_names.append["distortioner"]

    def forward(self, img: torch.FloatTensor, msg: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
        enc_img = self.encoder(img, msg)
        dis_img = self.distortioner(img, enc_img)
        pred_msg = self.decoder(dis_img)
        return enc_img, dis_img, pred_msg

    def parallel(self, device_ids: typing.List[int]):
        self._parallel(device_ids)


class Discriminator(_Base):
    def __init__(self):
        super().__init__()
        self.discriminator = pkg.architecture.Discriminator()
        self.module_names = ["discriminator"]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.discriminator(x)

    def parallel(self, device_ids: typing.List[int]):
        self._parallel(device_ids)
