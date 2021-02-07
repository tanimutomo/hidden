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

    def train(self, img, msg):
        raise NotImplementedError

    def test(self, img, msg):
        raise NotImplementedError

    def get_checkpoint(self) -> dict:
        raise NotImplementedError

    def load_checkpoint(self, ckpt: dict) -> dict:
        raise NotImplementedError


@dataclass
class HiddenTrainer(Trainer):

    device: torch.device
    gpu_ids: typing.List[int]
    model: torch.nn.Module
    ckpt: typing.Dict[str, object] ={}

    lambda_i: float =0.7
    lambda_g: float =0.001
    discriminator_lr: float =1e-3
    optimizer_lr: float =1e-3
    optimizer_wd: float =0

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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.optimizer_lr,
            weight_decay=self.optimizer_wd,
        )
        self.discriminator = Discriminator()
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator_lr)

        self._load_checkpoint()
        self._to_device()

    def train(self, img, msg) -> typing.Tuple[typing.Dict, typing.Dict]:
        self.discriminator.train()

        img, msg = img.to(self.device), msg.to(self.device)

        self.discriminator.zero_grad()
        label = torch.full((img.shape[0],), 1, dtype=torch.float, device=self.device)
        output = self.discriminator(img).view(-1)
        err_d_real = F.binary_cross_entropy(output, label)
        err_d_real.backward()

        enc_img, pred_msg = self.model(img, msg)
        label.fill_(0)
        err_d_fake = F.binary_cross_entropy(self.discriminator(enc_img.detach()).view(-1), label)
        err_d_fake.backward()

        self.discriminator_optimizer.step()

        self.model.zero_grad()
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

    def test(self, img, msg) -> typing.Tuple[typing.Dict, typing.Dict]:
        self.discriminator.eval()

        img, msg = img.to(self.device), msg.to(self.device)
        enc_img, pred_msg = self.model(img, msg)

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

    def get_checkpoint(self):
        return {
            "model": self.model_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
        }

    def model_state_dict(self) -> dict:
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _load_checkpoint(self):
        self.model.load_state_dict(self.ckpt["model"])
        self.optimizer.load_state_dict(self.ckpt["optimizer"])
        self.discriminator.load_state_dict(self.ckpt["discriminator"])
        self.discriminator_optimizer.load_state_dict(self.ckpt["discriminator_optimizer"])

    def _to_device(self):
        self.model.to(self.device)
        self.discriminator.to(self.device)
        if len(self.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)

