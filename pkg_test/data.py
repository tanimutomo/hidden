import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath("."))

from pkg.data import (
    COCOImageDataset,
)


class TestCOCOImageDataset(unittest.TestCase):
    """test for COCOImageDataset
    """

    def setUp(self):
        self.dataset = COCOImageDataset("./pkg_test/data/dataset")

    def test_len(self):
        """test method for forward
        """
        self.assertEqual(1, len(self.dataset))

    def test_getitem(self):
        """test method for getitem
        """
        self.assertIsNotNone(self.dataset[0])


if __name__ == "__main__":
    unittest.main()
