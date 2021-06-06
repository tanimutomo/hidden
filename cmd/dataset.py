import glob
import os

import hydra
import PIL
from torchvision import transforms


@hydra.main(config_name="config/dataset.yaml")
def main(cfg):
    orgs = [cfg.original_dataset.train_path, cfg.original_dataset.test_path]
    tars = [cfg.target_dataset.train_path, cfg.target_dataset.test_path]
    max_cnts = [10000, 1000]

    for org, tar, max_cnt in zip(orgs, tars, max_cnts):
        os.makedirs(tar, exist_ok=True)
        for i, f in enumerate(glob.glob(f"{org}/*")):
            img = PIL.Image.open(f)
            img = transforms.Compose([
                transforms.Resize(cfg.image_size),
                transforms.CenterCrop(cfg.image_size),
            ])(img)
            img.save(f"{tar}/{os.path.basename(f)}")
            if i+1 == max_cnt: break


if __name__ == "__main__":
    main()
