from dataclasses import dataclass
import os
import sys
import typing

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from pkg.meter import (
    MultiAverageMeter,
)
from pkg.loss import (
    Loss
)


class TrainManagerConfig(typing.NamedTuple):
    epochs: int =100
    test_interval: int =10


class TrainManager(object):
    def __init__(self, cfg: TrainManagerConfig, model, experiment):
        self.cfg = cfg
        self.model = model
        # self.experiment = experiment

    def train(self, train_loader, test_loader, optimizer):
        print("Start Training...")

        for epoch in self.cfg.epochs:
            # self.experiment.epoch = epoch
            self.meter = MultiAverageMeter(optimizer.loss_names)

            self.model.train()
            for step, (img, msg) in enumerate(train_loader):
                img, msg = img.to(self.device), msg.to(self.device)
                loss_dict, img_dict = optimizer(self.model, img, msg)
                self.meter.updates(loss_dict)
                # self.experiment.report("train", loss_dict)

            # self.experiment.save_image(img_dict, f"train_{itr}.png")

            if step % self.cfg.test_interval == 0:
                loss_dict, img_dict = self.test(test_loader, optimizer)
                # self.experiment.report("test", loss_dict)
                # self.experiment.save_image(save_inp, f"test_{itr}.png")

            # self.experiment.save_ckpt(self.model, self.optimizer)

    def test(self, test_loader, optimizer) -> typing.Tuple[dict, dict]:
        meter = MultiAverageMeter(optimizer.loss_names)
        self.model.eval()

        with torch.no_grad():
            with tqdm(test_loader, ncols=80, leave=False) as pbar:
                for itr, (img, msg) in enumerate(pbar):
                    img, msg = img.to(self.device), msg.to(self.device)
                    loss_dict, img_dict = optimizer(self.model, img, msg)
                    meter.updates(loss_dict())

                    pbar.set_postfix_str(f'loss={loss_dict["total"]:.4f}')

        return meter.to_dict, img_dict
