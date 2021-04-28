import typing
from dataclasses import dataclass

import torch

import pkg.dataset
import pkg.metric
import pkg.model


StateDict = typing.Dict[str, typing.Any]

class Cycle:

    model: pkg.model.HiddenModel
    metric_keys = []
    img_keys = []

    def train(self, item: pkg.dataset.BatchItem):
        raise NotImplementedError

    def test(self, item: pkg.dataset.BatchItem):
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
    optimizer_lr: float =1e-3
    optimizer_wd: float =0
    discriminator_lr: float =1e-3


@dataclass
class HiddenCycle(Cycle):

    loss_cfg: HiddenLossConfig
    model: pkg.model.HiddenModel

    device: torch.device
    gpu_ids: typing.List[int]

    wvec: pkg.wordvec.GloVe =None

    metric_keys = [
        "message_loss",
        "reconstruction_loss",
        "adversarial_generator_loss",
        "adversarial_discriminator_loss",
        "model_loss",
        "message_accuracy",
        "whole_message_accuracy",
    ]
    img_keys = [
        "input",
        "encoded",
        "noised",
    ]

    def __post_init__(self):
        self.discriminator = pkg.model.Discriminator()

    def setup_train(self, cfg: HiddenTrainConfig, ckpt: typing.Dict[str, object]):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.optimizer_lr,
            weight_decay=cfg.optimizer_wd,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=cfg.discriminator_lr)

        if ckpt:
            self._load_checkpoint(ckpt)
            _optimizer_to_device(self.optimizer, self.device)
            _optimizer_to_device(self.discriminator_optimizer, self.device)
        else:
            self.model.encoder.apply(_dcgan_weights_init)
            self.discriminator.apply(_dcgan_weights_init)

        self._setup()

    def setup_test(self, params: typing.Dict[str, object]):
        if params:
            self._load_parameters(params)
        self._setup()

    def _setup(self):
        self.model.to(self.device)
        self.model.parallel(self.gpu_ids)
        self.discriminator.to(self.device)
        self.discriminator.parallel(self.gpu_ids)

    def train(self, item: pkg.dataset.BatchItem) -> typing.Tuple[typing.Dict, typing.Dict]:
        self.discriminator.train()

        img, msg = item.img.to(self.device), item.msg_vec().to(self.device)

        self.discriminator.zero_grad()

        err_d_real = pkg.metric.adversarial_real_loss(self.discriminator, img, self.device)
        err_d_real.backward()

        enc_img, nos_img, pred_msg = self.model(img, msg)

        err_d_fake = pkg.metric.adversarial_fake_loss(self.discriminator, enc_img, self.device)
        err_d_fake.backward()

        self.discriminator_optimizer.step()

        self.model.zero_grad()

        err_g = pkg.metric.adversarial_generator_loss(self.discriminator, enc_img, self.device)
        err_msg = pkg.metric.message_loss(pred_msg, msg)
        err_rec = pkg.metric.reconstruction_loss(enc_img, img)

        err_model = err_msg + self.loss_cfg.lambda_i*err_rec + self.loss_cfg.lambda_g*err_g
        err_model.backward()

        self.optimizer.step()

        if item.is_msg_tensor():
            acc_msg = pkg.metric.message_accuracy(pred_msg, msg)
            wacc_msg = pkg.mertic.whole_message_accuracy(pred_msg, msg)
        else:
            acc_msg = pkg.metric.word_vector_accuracy(self.wvec, pred_msg, item.msg)
            wacc_msg = torch.tensor(0.0)

        metrics = {
            "message_loss": err_msg.item(),
            "reconstruction_loss": err_rec.item(),
            "adversarial_generator_loss": err_g.item(),
            "adversarial_discriminator_loss": err_d_real.item() + err_d_fake.item(),
            "model_loss": err_model.item(),
            "message_accuracy": acc_msg.item(),
            "whole_message_accuracy": wacc_msg.item(),
        }
        imgs = {
            "input": img[0].cpu().detach(),
            "encoded": enc_img[0].cpu().detach(),
            "noised": nos_img[0].cpu().detach(),
        }
        return metrics, imgs

    def test(self, item: pkg.dataset.BatchItem) -> typing.Tuple[typing.Dict, typing.Dict]:
        self.discriminator.eval()

        img, msg = item.img.to(self.device), item.msg_vec().to(self.device)
        enc_img, nos_img, pred_msg = self.model(img, msg)

        err_d_real = pkg.metric.adversarial_real_loss(self.discriminator, img, self.device)
        err_d_fake = pkg.metric.adversarial_fake_loss(self.discriminator, enc_img, self.device)

        err_g = pkg.metric.adversarial_generator_loss(self.discriminator, enc_img, self.device)
        err_msg = pkg.metric.message_loss(pred_msg, msg)
        err_rec = pkg.metric.reconstruction_loss(enc_img, img)

        err_model = err_msg + self.loss_cfg.lambda_i*err_rec + self.loss_cfg.lambda_g*err_g

        if item.is_msg_tensor():
            acc_msg = pkg.metric.message_accuracy(pred_msg, msg)
            wacc_msg = pkg.mertic.whole_message_accuracy(pred_msg, msg)
        else:
            acc_msg = pkg.metric.word_vector_accuracy(self.wvec, pred_msg, item.msg)
            wacc_msg = torch.tensor(0.0)

        metrics = {
            "message_loss": err_msg.item(),
            "reconstruction_loss": err_rec.item(),
            "adversarial_generator_loss": err_g.item(),
            "adversarial_discriminator_loss": err_d_real.item() + err_d_fake.item(),
            "model_loss": err_model.item(),
            "message_accuracy": acc_msg.item(),
            "whole_message_accuracy": wacc_msg.item(),
        }
        imgs = {
            "input": img[0].cpu().detach(),
            "encoded": enc_img[0].cpu().detach(),
            "noised": nos_img[0].cpu().detach(),
        }
        return metrics, imgs

    def get_checkpoint(self) -> StateDict:
        return {
            "model": self.model.state_dict(),
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
            "model": self.model.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def _load_parameters(self, params: StateDict):
        self.model.load_state_dict(params["model"])
        self.discriminator.load_state_dict(params["discriminator"])


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _model_state_dict(model: pkg.model.HiddenModel) -> dict:
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _model_to_device(model: pkg.model.HiddenModel, device: torch.device, device_ids: typing.List[int]) -> torch.nn.Module:
    model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def _dcgan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

