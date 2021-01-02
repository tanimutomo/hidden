from dataclasses import dataclass
import typing

import torch
import torch.nn as nn


@dataclass
class Encoder(nn.Module):
    inp_c: int =3
    out_c: int =3
    hid_c: int =64
    msg_c: int =30
    
    def __post_init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            ConvBlock(self.inp_c, self.hid_c),
            ConvBlock(self.hid_c, self.hid_c),
            ConvBlock(self.hid_c, self.hid_c),
            ConvBlock(self.hid_c, self.hid_c),
        )
        self.enc2 = nn.Sequential(
            ConvBlock(self.hid_c + self.msg_c + self.inp_c, self.hid_c),
            nn.Conv2d(self.hid_c, self.out_c, 1, 1, 0),
        )

    def forward(self, x: torch.FloatTensor, m: torch.FloatTensor) -> torch.FloatTensor:
        m = m[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        y = torch.cat([self.enc1(x), m], dim=1)
        return torch.enc2(y)


@dataclass
class Decoder(nn.Module):
    inp_c: int =3
    out_c: int =30
    hid_c: int =64

    def __post_init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            ConvBlock(self.inp_c, self.hid_c),
            *[ConvBlock(self.hid_c, self.hid_c)]*5,
            ConvBlock(self.hid_c, self.out_c),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.dec(x)[:, :, 0, 0]


@dataclass
class Discriminator(nn.Module):
    inp_c: int =6
    out_c: int =2
    hid_c: int =64

    def __post_init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBlock(self.inp_c, self.hid_c),
            ConvBlock(self.hid_c, self.hid_c),
            ConvBlock(self.hid_c, self.hid_c),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.hid_c, self.out_c)

    def forward(self, x1: torch.FloatTensor, x2: torch.FloatTensor) -> torch.FloatTensor:
        y = self.convs(torch.cat([x1, x2], dim=1))
        y = self.pool(y)[:, :, 0, 0]
        return self.fc(y)


@dataclass
class ConvBlock(nn.Module):
    inp_c: int
    out_c: int
    kernel_size: int =3
    stride: int =1
    padding: int =1

    def __post_init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(self.inp_c, self.out_c, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_c),
            nn.ReLU(),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.block(x)
