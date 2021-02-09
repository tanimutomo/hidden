import os
import sys
import typing

import distortion
import torch
import torch.nn as nn

sys.path.append(os.path.abspath("."))

from pkg.architecture import (
    Encoder,
    Decoder,
)


class HiddenModel(nn.Module):
    def __init__(self, distortioner: distortion.Distortioner):
        super().__init__()
        self.encoder = Encoder()
        self.distortioner = distortioner
        self.decoder = Decoder()

    def forward(self, img: torch.FloatTensor, msg: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
        enc_img = self.encoder(img, msg)
        dis_img = self.distortioner(img, enc_img)
        pred_msg = self.decoder(dis_img)
        return enc_img, pred_msg
