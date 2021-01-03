import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath("."))

from pkg.data import (
    WatermarkDataset,
)

DATASET_PATH = "./pkg_test/testdata/data/dataset"

class TestWatermarkDataset(unittest.TestCase):
    """test for WatermarkDataset
    """

    def setUp(self):
        self.dataset = WatermarkDataset(DATASET_PATH, 30)

    def test_len(self):
        """test method for forward
        """
        self.assertEqual(1, len(self.dataset))

    def test_getitem(self):
        """test method for getitem
        """
        img, msg = self.dataset[0]
        self.assertIsNotNone(img)
        self.assertIsNotNone(msg)


if __name__ == "__main__":
    unittest.main()
