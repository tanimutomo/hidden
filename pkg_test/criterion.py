import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath("."))

from pkg.architecture import (
    Discriminator
)
from pkg.criterion import (
    L2Loss,
    AdversarialLoss,
) 

class TestL2Loss(unittest.TestCase):
    """test for L2Loss
    """

    def setUp(self):
        self.loss = L2Loss()

    def test_forward_vector(self):
        """test method for forward
        """
        output = torch.rand(1, 30)
        target = torch.rand(1, 30)
        y = self.loss(output, target)
        self.assertEqual(torch.Size([]), y.shape)


    def test_forward_image(self):
        """test method for forward
        """
        output = torch.rand(1, 3, 32, 32)
        target = torch.rand(1, 3, 32, 32)
        y = self.loss(output, target)
        self.assertEqual(torch.Size([]), y.shape)


class TestAdversarialLoss(unittest.TestCase):
    """test for AdversarialLoss
    """

    def test_generator_loss(self):
        """test method for generator_loss
        """
        l = AdversarialLoss(Discriminator())
        output = torch.rand(1, 3, 32, 32)
        y = l.generator_loss(output)
        self.assertEqual(torch.Size([1, 1]), y.shape)

    def test_disctiminator_loss(self):
        """test method for disctiminator_loss
        """
        l = AdversarialLoss(Discriminator())
        output = torch.rand(1, 3, 32, 32)
        target = torch.rand(1, 3, 32, 32)
        y = l.discriminator_loss(output, target)
        self.assertEqual(torch.Size([1, 1]), y.shape)


if __name__ == "__main__":
    unittest.main()
