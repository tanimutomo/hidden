import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

import architecture as arch


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output :torch.FloatTensor, target :torch.FloatTensor) -> torch.FloatTensor:
        return F.mse_loss(output, target)


class AdversarialLoss(nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.dis = discriminator

    def generator_loss(self, x :torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(-torch.log(self.dis(x)))

    def discriminator_real_loss(self, x :torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(-torch.log(self.dis(x)))

    def discriminator_fake_loss(self, x :torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(- torch.log(1 - self.dis(x)))
