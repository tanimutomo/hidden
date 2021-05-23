import typing

import kornia
import torch
import torch.nn.functional as F

import pkg.wordvec
import pkg.transform


REAL_LABEL = 1
FAKE_LABEL = 0


class _Base:
    def __init__(self, imgtf: pkg.transform.ImageTransformer):
        self.imgtf = imgtf

    def reconstruction_loss(self, pred_img, img):
        return F.mse_loss(pred_img, img)

    def adversarial_real_loss(self, discriminator, img, device):
        return self._adversarial_loss(discriminator, img.detach(), device, REAL_LABEL)

    def adversarial_fake_loss(self, discriminator, img, device):
        return self._adversarial_loss(discriminator, img.detach(), device, FAKE_LABEL)

    def adversarial_generator_loss(self, discriminator, img, device):
        return self._adversarial_loss(discriminator, img, device, REAL_LABEL)

    def _adversarial_loss(self, discriminator, img, device, label_value):
        label = torch.full((img.shape[0],), label_value, dtype=torch.float, device=device)
        output = discriminator(img).view(-1)
        return F.binary_cross_entropy(output, label)

    def psnr_y(self, base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._psnr(base, target, 0)

    def psnr_u(self, base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._psnr(base, target, 1)

    def psnr_v(self, base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._psnr(base, target, 2)

    def _psnr(self, base: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
        base, target = self.imgtf.psnr(base), self.imgtf.psnr(target)
        return kornia.losses.psnr(base[:, dim, ...], target[:, dim, ...], 1)


class BitMetrics(_Base):
    def __init__(self, imgtf: pkg.transform.ImageTransformer):
        super().__init__(imgtf)

    def message_loss(self, pred_msg, msg):
        return F.mse_loss(pred_msg, msg)

    def unit_message_accuracy(self, pred_msg, msg):
        return torch.mean(1 - (_err_count(pred_msg, msg) / msg.shape[1]))

    def message_accuracy(self, pred_msg, msg):
        return 1 - (torch.sum(torch.clamp(_err_count(pred_msg, msg), 0, 1)) / msg.shape[0])


class WordMetrics(_Base):
    def __init__(self, w2v: pkg.wordvec.GloVe, imgtf: pkg.transform.ImageTransformer):
        self.w2v = w2v
        super().__init__(imgtf)

    def message_loss(self, pred_msg, msg, dim):
        assert msg.shape[-1] % dim == 0
        loss = 0.0
        for i in range(msg.shape[-1]//dim):
            s, e = i*dim, (i+1)*dim
            loss += F.mse_loss(pred_msg[:, s:e], msg[:, s:e])
        return loss

    def unit_message_accuracy(self, pred_msg: torch.Tensor, target_msg: torch.Tensor) -> torch.Tensor: 
        pred_msg = self.w2v.most_similar(self.w2v.unserialize(pred_msg))
        pred_idx, target_idx = pred_msg.idx, target_msg.idx
        target_idx = target_idx.to(pred_idx.device)
        return torch.sum(pred_idx == target_idx) / (target_idx.shape[0] * target_idx.shape[1] * 1.0)

    def message_accuracy(self, pred_msg: torch.Tensor, target_msg: torch.Tensor) -> torch.Tensor: 
        pred_msg = self.w2v.most_similar(self.w2v.unserialize(pred_msg))
        pred_idx, target_idx = pred_msg.idx, target_msg.idx
        target_idx = target_idx.to(pred_idx.device)
        corr = torch.sum(pred_idx == target_idx, dim=1) == torch.sum(torch.ones_like(target_idx), dim=1)
        return torch.sum(corr) / (target_idx.shape[0]* 1.0)


def _err_count(pred_msg, msg):
    bin_pred_msg = torch.round(torch.clamp(pred_msg, 0, 1))
    return torch.sum(torch.abs(bin_pred_msg - msg), dim=1)
