import typing
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pkg.architecture import (
    Discriminator,
)


StateDict = typing.Dict[str, typing.Any]

class Cycle:

    model: torch.nn.Module
    loss_keys = []
    img_keys = []

    def train(self, img, msg):
        raise NotImplementedError

    def test(self, img, msg):
        raise NotImplementedError

    def get_checkpoint(self) -> dict:
        raise NotImplementedError

    def get_parameters(self) -> dict:
        raise NotImplementedError


@dataclass
class HiddenLossConfig:
    lambda_i: float =0.7
    lambda_g: float =0.001


@dataclass
class HiddenTrainConfig:
    discriminator_lr: float =1e-3
    optimizer_lr: float =1e-3
    optimizer_wd: float =0


@dataclass
class HiddenTestConfig:
    nothing: typing.Any


@dataclass
class HiddenCycle(Cycle):

    loss_cfg: HiddenLossConfig
    model: torch.nn.Module

    device: torch.device
    gpu_ids: typing.List[int]

    loss_keys = [
        "message",
        "reconstruction",
        "adversarial_generator",
        "adversarial_discriminator",
        "model",
    ]
    img_keys = [
        "train",
        "test",
    ]

    def __post_init__(self):
        self.discriminator = Discriminator()

    def setup_train(self, cfg: HiddenTrainConfig, ckpt: typing.Dict[str, object]):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.optimizer_lr,
            weight_decay=cfg.optimizer_wd,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=cfg.discriminator_lr)

        if ckpt:
            self._load_checkpoint(ckpt)

        self._setup()

    def setup_test(self, cfg: HiddenTestConfig, params: typing.Dict[str, object]):
        if params:
            self._load_parameters(params)
        self._setup()

    def _setup(self):
        self.model = _model_to_device(self.model, self.device, self.gpu_ids)
        self.discriminator.to(self.device)

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
        err_model = err_msg + self.loss_cfg.lambda_i*err_rec + self.loss_cfg.lambda_g*err_g
        err_model.backward()

        self.optimizer.step()

        metrics = {
            "message": err_msg.item(),
            "reconstruction": err_rec.item(),
            "adversarial_generator": err_g.item(),
            "adversarial_discriminator": err_d_real.item() + err_d_fake.item(),
            "model": err_model.item(),
        }
        imgs = {
            "train": torch.stack([enc_img[0], img[0]]).cpu().detach(),
        }
        return metrics, imgs

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
        err_model = err_msg + self.loss_cfg.lambda_i*err_rec + self.loss_cfg.lambda_g*err_g

        metrics = {
            "message": err_msg.item(),
            "reconstruction": err_rec.item(),
            "adversarial_generator": err_g.item(),
            "adversarial_discriminator": err_d_real.item() + err_d_fake.item(),
            "model": err_model.item(),
        }
        imgs = {
            "test": torch.stack([enc_img[0], img[0]]).cpu().detach(),
        }
        return metrics, imgs

    def get_checkpoint(self) -> StateDict:
        return {
            "model": _model_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
        }

    def _load_checkpoint(self, ckpt: StateDict):
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.discriminator_optimizer.load_state_dict(ckpt["discriminator_optimizer"])

    def get_parameters(self) -> StateDict:
        return {
            "model": _model_state_dict(self.model),
            "discriminator": self.discriminator.state_dict(),
        }

    def _load_parameters(self, params: StateDict):
        self.model.load_state_dict(params["model"])
        self.discriminator.load_state_dict(params["discriminator"])


def _model_state_dict(model: torch.nn.Module) -> dict:
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _model_to_device(model: torch.nn.Module, device: torch.device, device_ids: typing.List[int]) -> torch.nn.Module:
    model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model
