import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

import pkg.architecture as arch


@dataclass
class MessageLoss(nn.Module):
    def forward(self, output :torch.FloatTensor, target :torch.FloatTensor) -> torch.FloatTensor:
        return F.mse_loss(output, target)


@dataclass
class ImageReconstructionLoss(nn.Module):
    def forward(self, output :torch.FloatTensor, target :torch.FloatTensor) -> torch.FloatTensor:
        return F.mse_loss(output, target)


@dataclass
class AdversarialLoss(nn.Module):
    def __post_init__(self):
        self.dis = arch.Discriminator()

    def generator_loss(self, output :torch.FloatTensor, target :torch.FloatTensor) -> torch.FloatTensor:
        return -torch.log(self.dis(output))

    def discriminator_loss(self, output :torch.FloatTensor, target :torch.FloatTensor) -> torch.FloatTensor:
        return -torch.log(self.dis(target)) - torch.log(1 - self.dis(output))

