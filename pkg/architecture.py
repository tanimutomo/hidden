from dataclasses import dataclass
import typing

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, inp_c: int =3, out_c: int =3, hid_c: int =64, msg_c: int =30):
        super().__init__()

        self.inp_c = inp_c
        self.out_c = out_c
        self.hid_c = hid_c
        self.msg_c = msg_c

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
        y = torch.cat([self.enc1(x), m, x], dim=1)
        return self.enc2(y)


class Decoder(nn.Module):
    def __init__(self, inp_c: int =3, out_c: int =30, hid_c: int =64):
        super().__init__()
        
        self.inp_c = inp_c
        self.out_c = out_c
        self.hid_c = hid_c

        self.dec = nn.Sequential(
            ConvBlock(self.inp_c, self.hid_c),
            *[ConvBlock(self.hid_c, self.hid_c)]*5,
            ConvBlock(self.hid_c, self.out_c),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.dec(x)[:, :, 0, 0]


class Discriminator(nn.Module):
    def __init__(self, inp_c: int =3, out_c: int =1, hid_c: int =64):
        super().__init__()

        self.inp_c = inp_c
        self.out_c = out_c
        self.hid_c = hid_c

        self.convs = nn.Sequential(
            ConvBlock(self.inp_c, self.hid_c),
            ConvBlock(self.hid_c, self.hid_c),
            ConvBlock(self.hid_c, self.hid_c),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.hid_c, self.out_c)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        y = self.convs(x)
        y = self.pool(y)[:, :, 0, 0]
        return self.fc(y)


class ConvBlock(nn.Module):
    def __init__(self, inp_c: int, out_c: int, kernel_size: int =3, stride: int =1, padding: int =1):
        super().__init__()

        self.inp_c = inp_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.block = nn.Sequential(
            nn.Conv2d(self.inp_c, self.out_c, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_c),
            nn.ReLU(),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.block(x)
