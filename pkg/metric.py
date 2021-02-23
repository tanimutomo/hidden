import torch
import torch.nn.functional as F


REAL_LABEL = 1
FAKE_LABEL = 0


def message_loss(pred_msg, msg):
    return F.mse_loss(pred_msg, msg)


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
    bin_pred_msg = torch.round(torch.clamp(pred_msg, 0, 1))
    err_count = torch.sum(torch.abs(bin_pred_msg - msg), dim=1)
    return torch.mean(1 - (err_count / msg.shape[1]))

