from dataclasses import dataclass
import os
import sys
import unittest
import typing

import torch
import torch.nn as nn

sys.path.append(os.path.abspath("."))

from pkg.criterion import (
    L2Loss,
    AdversarialLoss,
)
from pkg.architecture import (
    Discriminator,
)

class Loss(object):
    def calcurate(self, enc_img, pred_msg, img, msg):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

    def to_item_dict(self) -> dict:
        item_dict = dict()
        for name, loss in self.losses.items():
            item_dict[name] = loss.item()
        return item_dict

    def _reset(self):
        self.losses = {name: 0.0 for name in self.names}


@dataclass
class HiddenLoss(Loss):
    lambda_i: float =0.7
    lambda_g: float =0.001
    discriminator_lr: float =1e-3

    names = [
        "message",
        "reconstruction",
        "adversarial_generator",
        "adversarial_discriminator",
        "total",
    ]

    def __post_init__(self):
        super().__init__()
        self.message_loss = L2Loss()
        self.image_reconstruction_loss = L2Loss()
        
        self.discriminator = Discriminator()
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        self.adversarial_loss = AdversarialLoss(self.discriminator)

        self._reset()

    def calcurate(self, enc_img, pred_msg, img, msg):
        self._reset()
        self.losses = {
            "message": self.message_loss(pred_msg, msg),
            "reconstruction": self.image_reconstruction_loss(enc_img, img),
            "adversarial_generator": self.adversarial_loss.generator_loss(enc_img),
            "adversarial_discriminator": self.adversarial_loss.discriminator_real_loss(img) + self.adversarial_loss.discriminator_fake_loss(enc_img),
        }
        self._calcurate_total()
        
    def backward(self):
        self.losses["total"].backward()

    def discriminator_optimize(self):
        self.discriminator.zero_grad()
        self.losses["adversarial_discriminator"].backward()
        self.discriminator_optimizer.step()

    def _calcurate_total(self) -> torch.FloatTensor:
        self.losses["total"] = self.losses["message"] \
             + self.lambda_i * self.losses["reconstruction"] \
             + self.lambda_g * self.losses["adversarial_generator"]

    def to(self, device):
        self.discriminator.to(device)
