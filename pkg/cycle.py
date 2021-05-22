import typing
from dataclasses import dataclass, asdict

import torch

import pkg.dataset
import pkg.metric
import pkg.model


StateDict = typing.Dict[str, typing.Any]

@dataclass
class LossConfig:
    lambda_i: float =0.7
    lambda_g: float =0.001

@dataclass
class TrainConfig:
    optimizer_lr: float =1e-3
    optimizer_wd: float =0
    discriminator_lr: float =1e-3

@dataclass
class BitMetricOutput:
    message_loss: float
    reconstruction_loss: float
    adversarial_generator_loss: float
    adversarial_discriminator_loss: float
    model_loss: float
    message_accuracy: float
    unit_message_accuracy: float
    psnr_y: float
    psnr_u: float
    psnr_v: float

    @classmethod
    def keys(cls) -> list:
        return list(cls.__annotations__.keys())

    def todict(self) -> dict:
        return asdict(self)

@dataclass
class ImageOutput:
    input: torch.Tensor
    encoded: torch.Tensor
    noised: torch.Tensor
    
    def todict(self) -> dict:
        return asdict(self)

@dataclass
class BitOutput:
    metric: BitMetricOutput
    image: ImageOutput

@dataclass
class WordTextOutput:
    input: str
    predicted: str

    def todict(self):
        return asdict(self)

@dataclass
class WordMetricOutput:
    message_loss: float
    reconstruction_loss: float
    adversarial_generator_loss: float
    adversarial_discriminator_loss: float
    model_loss: float
    unit_message_accuracy: float
    message_accuracy: float
    psnr_y: float
    psnr_u: float
    psnr_v: float

    @classmethod
    def keys(cls) -> list:
        return list(cls.__annotations__.keys())

    def todict(self):
        return asdict(self)

@dataclass
class WordOutput:
    metric: WordMetricOutput
    image: ImageOutput
    text: WordTextOutput


class Cycle:
    model: pkg.model.HiddenModel

    def __init__(self, model: torch.nn.Module, device: torch.device, gpu_ids: typing.List[int]):
        self.model = model
        self.device = device
        self.gpu_ids = gpu_ids

        self.discriminator = pkg.model.Discriminator()

    def train(self, item: pkg.dataset.BatchItem):
        raise NotImplementedError()

    def test(self, item: pkg.dataset.BatchItem):
        raise NotImplementedError()

    def setup_train(self, cfg: TrainConfig, ckpt: typing.Dict[str, object]):
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



class BitCycle(Cycle):
    def __init__(self, loss_cfg: LossConfig, metrics: pkg.metric.BitMetrics, model: torch.nn.Module, device: torch.device, gpu_ids: typing.List[int]):
        self.loss_cfg = loss_cfg
        self.metrics = metrics

        super().__init__(model=model, device=device, gpu_ids=gpu_ids)

    def train(self, item: pkg.dataset.BatchItem) -> BitOutput:
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

        psnr_y = self.metrics.psnr_y(img, enc_img)
        psnr_u = self.metrics.psnr_u(img, enc_img)
        psnr_v = self.metrics.psnr_v(img, enc_img)

        return BitOutput(
            metric=BitMetricOutput(
                message_loss=err_msg.item(),
                reconstruction_loss=err_rec.item(),
                adversarial_generator_loss=err_g.item(),
                adversarial_discriminator_loss=err_d_real.item() + err_d_fake.item(),
                model_loss=err_model.item(),
                unit_message_accuracy=unit_acc_msg.item(),
                message_accuracy=acc_msg.item(),
                psnr_y=psnr_y.item(),
                psnr_u=psnr_u.item(),
                psnr_v=psnr_v.item(),
            ),
            image=ImageOutput(
                input=img[0].cpu().detach(),
                encoded=enc_img[0].cpu().detach(),
                noised=nos_img[0].cpu().detach(),
            ),
        )

    def test(self, item: pkg.dataset.BatchItem) -> BitOutput:
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

        psnr_y = self.metrics.psnr_y(img, enc_img)
        psnr_u = self.metrics.psnr_u(img, enc_img)
        psnr_v = self.metrics.psnr_v(img, enc_img)

        return BitOutput(
            metric=BitMetricOutput(
                message_loss=err_msg.item(),
                reconstruction_loss=err_rec.item(),
                adversarial_generator_loss=err_g.item(),
                adversarial_discriminator_loss=err_d_real.item() + err_d_fake.item(),
                model_loss=err_model.item(),
                unit_message_accuracy=unit_acc_msg.item(),
                message_accuracy=acc_msg.item(),
                psnr_y=psnr_y.item(),
                psnr_u=psnr_u.item(),
                psnr_v=psnr_v.item(),
            ),
            image=ImageOutput(
                input=img[0].cpu().detach(),
                encoded=enc_img[0].cpu().detach(),
                noised=nos_img[0].cpu().detach(),
            ),
        )


class WordCycle(Cycle):
    def __init__(self, loss_cfg: LossConfig, metrics: pkg.metric.BitMetrics, w2v: pkg.wordvec.GloVe, model: torch.nn.Module, device: torch.device, gpu_ids: typing.List[int]):
        self.loss_cfg = loss_cfg
        self.metrics = metrics
        self.w2v = w2v

        super().__init__(model=model, device=device, gpu_ids=gpu_ids)

    def train(self, item: pkg.dataset.BatchItem) -> WordOutput:
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

        psnr_y = self.metrics.psnr_y(img, enc_img)
        psnr_u = self.metrics.psnr_u(img, enc_img)
        psnr_v = self.metrics.psnr_v(img, enc_img)

        return WordOutput(
            metric=WordMetricOutput(
                message_loss=err_msg.item(),
                reconstruction_loss=err_rec.item(),
                adversarial_generator_loss=err_g.item(),
                adversarial_discriminator_loss=err_d_real.item() + err_d_fake.item(),
                model_loss=err_model.item(),
                unit_message_accuracy=unit_msg_acc.item(),
                message_accuracy=msg_acc.item(),
                psnr_y=psnr_y.item(),
                psnr_u=psnr_u.item(),
                psnr_v=psnr_v.item(),
            ),
            image=ImageOutput(
                input=img[0].cpu().detach(),
                encoded=enc_img[0].cpu().detach(),
                noised=nos_img[0].cpu().detach(),
            ),
            text=WordTextOutput(
                input=ipt_txt,
                predicted=pred_txt,
            )
        )

    def test(self, item: pkg.dataset.BatchItem) -> WordOutput:
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

        psnr_y = self.metrics.psnr_y(img, enc_img)
        psnr_u = self.metrics.psnr_u(img, enc_img)
        psnr_v = self.metrics.psnr_v(img, enc_img)

        return WordOutput(
            metric=WordMetricOutput(
                message_loss=err_msg.item(),
                reconstruction_loss=err_rec.item(),
                adversarial_generator_loss=err_g.item(),
                adversarial_discriminator_loss=err_d_real.item() + err_d_fake.item(),
                model_loss=err_model.item(),
                unit_message_accuracy=unit_msg_acc.item(),
                message_accuracy=msg_acc.item(),
                psnr_y=psnr_y.item(),
                psnr_u=psnr_u.item(),
                psnr_v=psnr_v.item(),
            ),
            image=ImageOutput(
                input=img[0].cpu().detach(),
                encoded=enc_img[0].cpu().detach(),
                noised=nos_img[0].cpu().detach(),
            ),
            text=WordTextOutput(
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

