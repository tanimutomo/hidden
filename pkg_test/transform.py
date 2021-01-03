import os
import sys
import unittest

import PIL as pil
import torch

sys.path.append(os.path.abspath("."))

from pkg.transform import (
    Transformer
)

SAMPLE_IMAGE_PATH = "./pkg_test/testdata/transform/sample.jpg"

class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.transformer = Transformer()

    def test_train_image_transform(self):
        t = self.transformer.train_image_transform()
        img = pil.Image.open(SAMPLE_IMAGE_PATH)
        transformed = t(img)
        self.assertEqual(torch.Size([3, 128, 128]), transformed.shape)

    def test_test_image_transform(self):
        t = self.transformer.test_image_transform()
        img = pil.Image.open(SAMPLE_IMAGE_PATH)
        transformed = t(img)
        self.assertEqual(torch.Size([3, 128, 128]), transformed.shape)


if __name__ == "__main__":
    unittest.main()
