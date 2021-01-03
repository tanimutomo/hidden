import unittest

import torch

from loss import (
    MessageLoss,
    ImageReconstructionLoss,
    AdversarialLoss,
)


class TestMessageLoss(unittest.TestCase):
    """test for MessageLoss
    """

    def test_forward(self):
        """test method for forward
        """
        f = MessageLoss()
        output = torch.rand(1, 30)
        target = torch.rand(1, 30)
        y = f(output, target)
        self.assertEqual(torch.Size([]), y.shape)


class TestImageReconstructionLoss(unittest.TestCase):
    """test for ImageReconstructionLoss
    """

    def test_forward(self):
        """test method for forward
        """
        f = ImageReconstructionLoss()
        output = torch.rand(1, 3, 32, 32)
        target = torch.rand(1, 3, 32, 32)
        y = f(output, target)
        self.assertEqual(torch.Size([]), y.shape)


class TestAdversarialLoss(unittest.TestCase):
    """test for AdversarialLoss
    """

    def test_generator_loss(self):
        """test method for generator_loss
        """
        l = AdversarialLoss()
        output = torch.rand(1, 3, 32, 32)
        y = l.generator_loss(output)
        self.assertEqual(torch.Size([1, 1]), y.shape)

    def test_disctiminator_loss(self):
        """test method for disctiminator_loss
        """
        l = AdversarialLoss()
        output = torch.rand(1, 3, 32, 32)
        target = torch.rand(1, 3, 32, 32)
        y = l.discriminator_loss(output, target)
        self.assertEqual(torch.Size([1, 1]), y.shape)


if __name__ == "__main__":
    unittest.main()
