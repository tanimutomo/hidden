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

@dataclass
class HiddenLoss(object):
    lambda_i: float =0.7
    lambda_g: float =0.001
    discriminator_lr: float =1e-3

    names: list[str] = [
        "message",
        "reconstruction",
        "adversarial_generator",
        "adversarial_discriminator",
        "total",
    ]

    def __post_init__(self):
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
            "reconstruction": self.image_reconstruction_loss(enc_img, img)
            "adversarial_generator": self.adversarial_loss.generator_loss(enc_img),
            "adversarial_discriminator": self.adversarial_loss.discriminator_loss(enc_img, img)
        }
        self.calcurate_total()
        
    def backward(self):
        self.losses["total"].backward()

    def discriminator_optimize(self) -> torch.FloatTensor:
        self.discriminator.zero_grad()
        self.losses["adversarial_discriminator"].backward()
        self.discriminator_optimizer.step()

    def to_dict(self) -> dict:
        self.losses

    def _calcurate_total(self) -> torch.FloatTensor:
        self.losses["total"] = self.losses["message"] \
            + self.lambda_i * self.reconstruction["reconstruction"] \
            + self.lambda_g * self.adversarial_generator["adversarial_generator"]

    def _reset(self):
        self.losses = {name: 0.0 for name in self.names}
