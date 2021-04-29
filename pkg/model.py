import os
import sys
import typing

import distortion
import torch
import torch.nn as nn

sys.path.append(os.path.abspath("."))

import pkg.architecture


class _Base(nn.Module):
    _is_parallel: bool =False

    def __init__(self, parallel_modules: typing.List[str], trainable_modules: typing.List[str]):
        super().__init__()
        self.parallel_modules = parallel_modules
        self.trainable_modules = trainable_modules

    def parallel(self, device_ids: typing.List[int]):
        if len(device_ids) <= 1: return
        for m in self.parallel_modules:
            setattr(self, m, torch.nn.DataParallel(getattr(self, m), device_ids=device_ids))
        self._is_parallel = True

    def state_dict(self):
        sd = dict()
        for m in self.trainable_modules:
            module = getattr(self, m)
            if m in self.parallel_modules and self._is_parallel:
                module = module.module
            for k, v in module.state_dict().items():
                sd[f"{m}.{k}"] = v
        return sd

    def load_state_dict(self, state_dict, strict: bool =True):
        for m in self.trainable_modules:
            sd = dict()
            for k, v in state_dict.items():
                if k[:len(m)] == m: sd[k[len(m)+1:]] = v
            getattr(self, m).load_state_dict(sd, strict)


class HiddenModel(_Base):
    def __init__(
        self,
        msg_len: int,
        train_distorter: distortion.Distorter =None,
        test_distorter: distortion.Distorter =None,
        train_distortion_parallelable: bool =True,
        test_distortion_parallelable: bool =True,
    ):
        parallel_modules = ["encoder", "decoder"]
        if train_distortion_parallelable: parallel_modules.append("train_distorter")
        if test_distortion_parallelable: parallel_modules.append("test_distorter")
        super().__init__(
            parallel_modules=parallel_modules,
            trainable_modules=["encoder", "decoder"],
        )

        self.encoder = pkg.architecture.Encoder(msg_c=msg_len)
        self.train_distorter = train_distorter
        self.test_distorter = test_distorter
        self.decoder = pkg.architecture.Decoder(out_c=msg_len)

    def forward(self, img: torch.FloatTensor, msg: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
        enc_img = self.encoder(img, msg)
        dis = self.train_distorter if self.training else self.test_distorter
        dis_img = dis(img, enc_img)
        pred_msg = self.decoder(dis_img)
        return enc_img, dis_img, pred_msg


class Discriminator(_Base):
    def __init__(self):
        super().__init__(
            parallel_modules=["discriminator"],
            trainable_modules=["discriminator"],
        )
        self.discriminator = pkg.architecture.Discriminator()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.discriminator(x)
