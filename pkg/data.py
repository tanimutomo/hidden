import os
import io

import PIL as pil
import numpy as np
import torch
import torchvision


class COCOImageDataset(torch.utils.data.Dataset):
    """COCO Image Dataset"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return sum(os.path.isfile(os.path.join(self.root_dir, name)) for name in os.listdir(self.root_dir))

    def __getitem__(self, idx: int):
        img = pil.Image.open(os.path.join(self.root_dir, self.files[idx]))
        if self.transform:
            return self.transform(img)
        return img

