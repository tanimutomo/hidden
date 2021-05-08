import typing

import torch
import torch.nn.functional as F

import pkg.wordvec


REAL_LABEL = 1
FAKE_LABEL = 0


def message_loss(pred_msg, msg):
    return F.mse_loss(pred_msg, msg)


def word_message_loss(pred_msg, msg, dim):
    assert msg.shape[-1] % dim == 0
    loss = 0.0
    for i in range(msg.shape[-1]//dim):
        loss += F.mse_loss(pred_msg[:, i:i+dim], msg[:, i:i+dim])
    return loss


def reconstruction_loss(pred_img, img):
    return F.mse_loss(pred_img, img)


def adversarial_real_loss(discriminator, img, device):
    return _adversarial_loss(discriminator, img.detach(), device, REAL_LABEL)


def adversarial_fake_loss(discriminator, img, device):
    return _adversarial_loss(discriminator, img.detach(), device, FAKE_LABEL)


def adversarial_generator_loss(discriminator, img, device):
    return _adversarial_loss(discriminator, img, device, REAL_LABEL)


def _adversarial_loss(discriminator, img, device, label_value):
    label = torch.full((img.shape[0],), label_value, dtype=torch.float, device=device)
    output = discriminator(img).view(-1)
    return F.binary_cross_entropy(output, label)


def message_accuracy(pred_msg, msg):
    return torch.mean(1 - (_err_count(pred_msg, msg) / msg.shape[1]))


def whole_message_accuracy(pred_msg, msg):
    return 1 - (torch.sum(torch.clamp(_err_count(pred_msg, msg), 0, 1)) / msg.shape[0])


def _err_count(pred_msg, msg):
    bin_pred_msg = torch.round(torch.clamp(pred_msg, 0, 1))
    return torch.sum(torch.abs(bin_pred_msg - msg), dim=1)


def zero(_1, _2):
    return torch.tensor(0.0)


def word_message_accuracy(pred_idx: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor: 
    target_idx = target_idx.to(pred_idx.device)
    return torch.sum(pred_idx == target_idx) / (target_idx.shape[0] * 1.0)
