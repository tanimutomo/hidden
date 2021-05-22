import typing
from dataclasses import dataclass, asdict

import torch

import pkg.dataset
import pkg.metric
import pkg.model


StateDict = typing.Dict[str, typing.Any]

class Output:
    pass

class Cycle:

    model: pkg.model.HiddenModel

    def train(self, item: pkg.dataset.BatchItem) -> Output:
        raise NotImplementedError

    def test(self, item: pkg.dataset.BatchItem) -> Output:
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
class HiddenMetricOutput:
    message_loss: float
    reconstruction_loss: float
    adversarial_generator_loss: float
    adversarial_discriminator_loss: float
    model_loss: float
    message_accuracy: float
    unit_message_accuracy: float

    @classmethod
    def keys(cls) -> list:
        return list(cls.__annotations__.keys())

    def todict(self) -> dict:
        return asdict(self)

@dataclass
class HiddenImageOutput:
    input: torch.Tensor
    encoded: torch.Tensor
    noised: torch.Tensor
    
    def todict(self) -> dict:
        return asdict(self)

@dataclass
class HiddenOutput:
    metric: HiddenMetricOutput
    image: HiddenImageOutput

@dataclass
class WordHiddenTextOutput:
    input: str
    predicted: str

    def todict(self):
        return asdict(self)

@dataclass
class WordHiddenMetricOutput:
    message_loss: float
    reconstruction_loss: float
    adversarial_generator_loss: float
    adversarial_discriminator_loss: float
    model_loss: float
    unit_message_accuracy: float
    message_accuracy: float

    @classmethod
    def keys(cls) -> list:
        return list(cls.__annotations__.keys())

    def todict(self):
        return asdict(self)

@dataclass
class WordHiddenOutput:
    metric: WordHiddenMetricOutput
    image: HiddenImageOutput
    text: WordHiddenTextOutput


@dataclass
class HiddenCycle(Cycle):

    loss_cfg: HiddenLossConfig
    model: pkg.model.HiddenModel
    metrics: pkg.metric.BitMetrics

    device: torch.device
    gpu_ids: typing.List[int]

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

    def train(self, item: pkg.dataset.BatchItem) -> HiddenOutput:
        self.discriminator.train()

        item.to_(self.device)
        img, msg = item.img(), item.msg_vec()

        self.discriminator.zero_grad()

        err_d_real = self.metrics.adversarial_real_loss(self.discriminator, img, self.device)
        err_d_real.backward()

        enc_img, nos_img, pred_msg = self.model(img, msg)

        err_d_fake = self.metrics.adversarial_fake_loss(self.discriminator, enc_img, self.device)
        err_d_fake.backward()

        self.discriminator_optimizer.step()

        self.model.zero_grad()

        err_g = self.metrics.adversarial_generator_loss(self.discriminator, enc_img, self.device)
        err_msg = self.metrics.message_loss(pred_msg, msg)
        err_rec = self.metrics.reconstruction_loss(enc_img, img)

        err_model = err_msg + self.loss_cfg.lambda_i*err_rec + self.loss_cfg.lambda_g*err_g
        err_model.backward()

        self.optimizer.step()

        acc_msg = self.metrics.message_accuracy(pred_msg, msg)
        unit_acc_msg = self.metrics.unit_message_accuracy(pred_msg, msg)

        return HiddenOutput(
            metric=HiddenMetricOutput(
                message_loss=err_msg.item(),
                reconstruction_loss=err_rec.item(),
                adversarial_generator_loss=err_g.item(),
                adversarial_discriminator_loss=err_d_real.item() + err_d_fake.item(),
                model_loss=err_model.item(),
                unit_message_accuracy=unit_acc_msg.item(),
                message_accuracy=acc_msg.item(),
            ),
            image=HiddenImageOutput(
                input=img[0].cpu().detach(),
                encoded=enc_img[0].cpu().detach(),
                noised=nos_img[0].cpu().detach(),
            ),
        )

    def test(self, item: pkg.dataset.BatchItem) -> HiddenOutput:
        self.discriminator.eval()

        item.to_(self.device)
        img, msg = item.img().to(self.device), item.msg_vec().to(self.device)
        enc_img, nos_img, pred_msg = self.model(img, msg)

        err_d_real = self.metrics.adversarial_real_loss(self.discriminator, img, self.device)
        err_d_fake = self.metrics.adversarial_fake_loss(self.discriminator, enc_img, self.device)

        err_g = self.metrics.adversarial_generator_loss(self.discriminator, enc_img, self.device)
        err_msg = self.metrics.message_loss(pred_msg, msg)
        err_rec = self.metrics.reconstruction_loss(enc_img, img)

        err_model = err_msg + self.loss_cfg.lambda_i*err_rec + self.loss_cfg.lambda_g*err_g

        acc_msg = self.metrics.message_accuracy(pred_msg, msg)
        unit_acc_msg = self.metrics.unit_message_accuracy(pred_msg, msg)

        return HiddenOutput(
            metric=HiddenMetricOutput(
                message_loss=err_msg.item(),
                reconstruction_loss=err_rec.item(),
                adversarial_generator_loss=err_g.item(),
                adversarial_discriminator_loss=err_d_real.item() + err_d_fake.item(),
                model_loss=err_model.item(),
                unit_message_accuracy=unit_acc_msg.item(),
                message_accuracy=acc_msg.item(),
            ),
            image=HiddenImageOutput(
                input=img[0].cpu().detach(),
                encoded=enc_img[0].cpu().detach(),
                noised=nos_img[0].cpu().detach(),
            ),
        )

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


class WordHiddenCycle(HiddenCycle):
    def __init__(self, loss_cfg, model, metrics, w2v, device, gpu_ids):
        self.loss_cfg = loss_cfg
        self.model = model
        self.metrics = metrics
        self.w2v = w2v
        self.device = device
        self.gpu_ids = gpu_ids

        super().__init__(self.loss_cfg, self.model, self.metrics, self.device, self.gpu_ids)

    def train(self, item: pkg.dataset.BatchItem) -> WordHiddenOutput:
        self.discriminator.train()

        item.to_(self.device)
        img, msg = item.img(), item.msg_vec()

        self.discriminator.zero_grad()

        err_d_real = self.metrics.adversarial_real_loss(self.discriminator, img, self.device)
        err_d_real.backward()

        enc_img, nos_img, pred_msg = self.model(img, msg)

        err_d_fake = self.metrics.adversarial_fake_loss(self.discriminator, enc_img, self.device)
        err_d_fake.backward()

        self.discriminator_optimizer.step()

        self.model.zero_grad()

        err_g = self.metrics.adversarial_generator_loss(self.discriminator, enc_img, self.device)
        err_msg = self.metrics.message_loss(pred_msg, msg, self.w2v.dim)
        err_rec = self.metrics.reconstruction_loss(enc_img, img)

        err_model = err_msg + self.loss_cfg.lambda_i*err_rec + self.loss_cfg.lambda_g*err_g
        err_model.backward()

        self.optimizer.step()

        msg_acc = self.metrics.message_accuracy(pred_msg, item.msg())
        unit_msg_acc = self.metrics.unit_message_accuracy(pred_msg, item.msg())

        ipt_txt = self.w2v.get_keys(item.msg().idx[0])
        pred_txt = self.w2v.get_keys(self.w2v.most_similar(self.w2v.unserialize(pred_msg)).idx[0])

        return WordHiddenOutput(
            metric=WordHiddenMetricOutput(
                message_loss=err_msg.item(),
                reconstruction_loss=err_rec.item(),
                adversarial_generator_loss=err_g.item(),
                adversarial_discriminator_loss=err_d_real.item() + err_d_fake.item(),
                model_loss=err_model.item(),
                unit_message_accuracy=unit_msg_acc.item(),
                message_accuracy=msg_acc.item(),
            ),
            image=HiddenImageOutput(
                input=img[0].cpu().detach(),
                encoded=enc_img[0].cpu().detach(),
                noised=nos_img[0].cpu().detach(),
            ),
            text=WordHiddenTextOutput(
                input=ipt_txt,
                predicted=pred_txt,
            )
        )

    def test(self, item: pkg.dataset.BatchItem) -> WordHiddenOutput:
        self.discriminator.eval()

        item.to_(self.device)
        img, msg = item.img().to(self.device), item.msg_vec().to(self.device)
        enc_img, nos_img, pred_msg = self.model(img, msg)

        err_d_real = self.metrics.adversarial_real_loss(self.discriminator, img, self.device)
        err_d_fake = self.metrics.adversarial_fake_loss(self.discriminator, enc_img, self.device)

        err_g = self.metrics.adversarial_generator_loss(self.discriminator, enc_img, self.device)
        err_msg = self.metrics.message_loss(pred_msg, msg, self.w2v.dim)
        err_rec = self.metrics.reconstruction_loss(enc_img, img)

        err_model = err_msg + self.loss_cfg.lambda_i*err_rec + self.loss_cfg.lambda_g*err_g

        msg_acc = self.metrics.message_accuracy(pred_msg, item.msg())
        unit_msg_acc = self.metrics.unit_message_accuracy(pred_msg, item.msg())

        ipt_txt = self.w2v.get_keys(item.msg().idx[0])
        pred_txt = self.w2v.get_keys(self.w2v.most_similar(self.w2v.unserialize(pred_msg)).idx[0])

        return WordHiddenOutput(
            metric=WordHiddenMetricOutput(
                message_loss=err_msg.item(),
                reconstruction_loss=err_rec.item(),
                adversarial_generator_loss=err_g.item(),
                adversarial_discriminator_loss=err_d_real.item() + err_d_fake.item(),
                model_loss=err_model.item(),
                unit_message_accuracy=unit_msg_acc.item(),
                message_accuracy=msg_acc.item(),
            ),
            image=HiddenImageOutput(
                input=img[0].cpu().detach(),
                encoded=enc_img[0].cpu().detach(),
                noised=nos_img[0].cpu().detach(),
            ),
            text=WordHiddenTextOutput(
                input=ipt_txt,
                predicted=pred_txt,
            )
        )


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

