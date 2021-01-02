import unittest

import torch

from model import (
    Encoder,
    Decoder,
    Discriminator,
    ConvBlock,
)


class TestConvBlock(unittest.TestCase):
    """test for ConvBlock
    """

    def test_forward(self):
        """test method for forward
        """
        block = ConvBlock(3, 64)
        x = torch.rand(1, 3, 32, 32)
        y = block(x)
        self.assertEqual(torch.Size([1, 64, 32, 32]), y.shape)


class TestEncoder(unittest.TestCase):
    """test for Encoder
    """

    def test_forward(self):
        """test method for forward
        """
        enc = Encoder()
        x = torch.rand(1, 3, 32, 32)
        m = torch.rand(1, 30)
        y = enc(x, m)
        self.assertEqual(torch.Size([1, 3, 32, 32]), y.shape)


class TestDecoder(unittest.TestCase):
    """test for Decoder
    """

    def test_forward(self):
        """test method for forward
        """
        dec = Decoder()
        x = torch.rand(1, 3, 32, 32)
        y = dec(x)
        self.assertEqual(torch.Size([1, 30]), y.shape)


class TestDiscriminator(unittest.TestCase):
    """test for Discriminator
    """

    def test_forward(self):
        """test method for forward
        """
        dis = Discriminator()
        x1, x2 = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
        y = dis(x1, x2)
        self.assertEqual(torch.Size([1, 2]), y.shape)


if __name__ == "__main__":
    unittest.main()
