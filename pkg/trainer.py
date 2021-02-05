import typing
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pkg.architecture import (
    Discriminator,
)


class Trainer:

    loss_keys = []
    img_keys = []

    def __init__(self):
        # abstract class
        pass

    def train(self, model, img, msg):
        raise NotImplementedError

    def test(self, model, img, msg):
        raise NotImplementedError


@dataclass
class HiddenTrainer(Trainer):

    device: torch.device
    optimizer: torch.optim.Optimizer

    lambda_i: float =0.7
    lambda_g: float =0.001
    discriminator_lr: float =1e-3

    loss_keys = [
        "message",
        "reconstruction",
        "adversarial_generator",
        "adversarial_discriminator",
        "model",
    ]
    img_keys = [
        "train",
    ]

    def __post_init__(self):
        self.discriminator = Discriminator().to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator_lr)

    def train(self, model, img, msg) -> typing.Tuple[typing.Dict, typing.Dict]:
        self.discriminator.train()

        img, msg = img.to(self.device), msg.to(self.device)

        self.discriminator.zero_grad()
        label = torch.full((img.shape[0],), 1, dtype=torch.float, device=self.device)
        output = self.discriminator(img).view(-1)
        err_d_real = F.binary_cross_entropy(output, label)
        err_d_real.backward()

        enc_img, pred_msg = model(img, msg)
        label.fill_(0)
        err_d_fake = F.binary_cross_entropy(self.discriminator(enc_img.detach()).view(-1), label)
        err_d_fake.backward()

        self.discriminator_optimizer.step()

        model.zero_grad()
        err_g = F.binary_cross_entropy(self.discriminator(enc_img.detach()).view(-1), label)
        err_msg = F.mse_loss(pred_msg, msg)
        err_rec = F.mse_loss(enc_img, img)
        err_model = err_msg + self.lambda_i*err_rec + self.lambda_g*err_g
        err_model.backward()

        self.optimizer.step()

        losses = {
            "message": err_msg.item(),
            "reconstruction": err_rec.item(),
            "adversarial_generator": err_g.item(),
            "adversarial_discriminator": err_d_real.item() + err_d_fake.item(),
            "model": err_model.item(),
        }
        imgs = {
            "train": torch.stack([enc_img[0], img[0]]).cpu().detach(),
        }
        return losses, imgs

    def test(self, model, img, msg) -> typing.Tuple[typing.Dict, typing.Dict]:
        self.discriminator.eval()

        img, msg = img.to(self.device), msg.to(self.device)
        enc_img, pred_msg = model(img, msg)

        label = torch.full((img.shape[0],), 1, dtype=torch.float, device=self.device)
        err_d_real = F.binary_cross_entropy(self.discriminator(img).view(-1), label)
        label.fill_(0)
        err_d_fake = F.binary_cross_entropy(self.discriminator(enc_img.detach()).view(-1), label)

        err_g = F.binary_cross_entropy(self.discriminator(enc_img.detach()).view(-1), label)
        err_msg = F.mse_loss(pred_msg, msg)
        err_rec = F.mse_loss(enc_img, img)
        err_model = err_msg + self.lambda_i*err_rec + self.lambda_g*err_g

        losses = {
            "message": err_msg.item(),
            "reconstruction": err_rec.item(),
            "adversarial_generator": err_g.item(),
            "adversarial_discriminator": err_d_real.item() + err_d_fake.item(),
            "model": err_model.item(),
        }
        imgs = {
            "train": torch.stack([enc_img[0], img[0]]).cpu().detach(),
        }
        return losses, imgs
