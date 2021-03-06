import os
import io

from PIL import Image
import torch


class WatermarkDataset(torch.utils.data.Dataset):
    """Watermark Image Dataset"""

    def __init__(self, root_dir, msg_len, img_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.msg_len = int(msg_len)
        self.img_transform = img_transform
        self.files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return sum(os.path.isfile(os.path.join(self.root_dir, name)) for name in os.listdir(self.root_dir))

    def __getitem__(self, idx: int):
        img = Image.open(os.path.join(self.root_dir, self.files[idx]))
        msg = torch.rand(self.msg_len).round()
        if self.img_transform:
            img = self.img_transform(img)
        return img, msg

