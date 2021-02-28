from dataclasses import dataclass
import typing

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, inp_c: int =3, out_c: int =3, hid_c: int =64, msg_c: int =30):
        super().__init__()

        self.img_enc = nn.Sequential(
            ConvBlock(inp_c, hid_c),
            ConvBlock(hid_c, hid_c),
            ConvBlock(hid_c, hid_c),
            ConvBlock(hid_c, hid_c),
        )
        self.post_enc = nn.Sequential(
            ConvBlock(hid_c + msg_c + inp_c, hid_c),
            nn.Conv2d(hid_c, out_c, 1, 1, 0),
        )

    def forward(self, img: torch.FloatTensor, msg: torch.FloatTensor) -> torch.FloatTensor:
        msg = msg[:, :, None, None].repeat(1, 1, img.shape[-2], img.shape[-1])
        hid_img = self.img_enc(img)
        return self.post_enc(torch.cat([hid_img, msg, img], dim=1))


class Decoder(nn.Module):
    def __init__(self, inp_c: int =3, out_c: int =30, hid_c: int =64):
        super().__init__()

        self.convs = nn.Sequential(
            ConvBlock(inp_c, hid_c),
            *[ConvBlock(hid_c, hid_c)]*5,
            ConvBlock(hid_c, out_c),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_c, out_c)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.convs(x)
        x = self.pool(x)[:, :, 0, 0]
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, inp_c: int =3, out_c: int =1, hid_c: int =64):
        super().__init__()

        self.convs = nn.Sequential(
            ConvBlock(inp_c, hid_c),
            ConvBlock(hid_c, hid_c),
            ConvBlock(hid_c, hid_c),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hid_c, out_c)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        y = self.convs(x)
        y = self.pool(y)[:, :, 0, 0]
        y = self.fc(y)
        y = self.sig(y)
        return y


class ConvBlock(nn.Module):
    def __init__(self, inp_c: int, out_c: int, kernel_size: int =3, stride: int =1, padding: int =1, bn_track_running_stats: bool =True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inp_c, out_c, kernel_size, stride, padding),
            nn.BatchNorm2d(out_c, track_running_stats=bn_track_running_stats),
            nn.ReLU(),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.block(x)
