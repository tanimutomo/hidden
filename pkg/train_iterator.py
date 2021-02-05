from dataclasses import dataclass
import os
import sys
import typing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from pkg.meter import (
    MultiAverageMeter,
)
from trainer import (
    Trainer,
)


class TrainIteratorConfig(typing.NamedTuple):
    epochs: int =100
    test_interval: int =10


class TrainIterator(object):
    def __init__(self, cfg: TrainIteratorConfig, model: torch.nn.Module, experiment):
        self.cfg = cfg
        self.model = model
        # self.experiment = experiment

    def train(self, train_loader: DataLoader, test_loader: DataLoader, trainer: Trainer):
        print("Start Training...")

        for epoch in range(self.cfg.epochs):
            # self.experiment.epoch = epoch
            self.meter = MultiAverageMeter(trainer.loss_keys)

            self.model.train()
            for step, (img, msg) in enumerate(train_loader):
                loss_dict, img_dict = trainer.train(self.model, img, msg)
                self.meter.updates(loss_dict)
                print(f"step: {step}\n\tloss: {loss_dict}")
                # self.experiment.report("train", loss_dict)

            # self.experiment.save_image(img_dict, f"train_{itr}.png")

            if step % self.cfg.test_interval == 0:
                loss_dict, img_dict = self.test(test_loader, trainer)
                # self.experiment.report("test", loss_dict)
                # self.experiment.save_image(save_inp, f"test_{itr}.png")

            # self.experiment.save_ckpt(self.model, self.optimizer)

    def test(self, test_loader: DataLoader, trainer: Trainer) -> typing.Tuple[dict, dict]:
        meter = MultiAverageMeter(trainer.loss_keys)
        self.model.eval()

        with torch.no_grad():
            with tqdm(test_loader, ncols=80, leave=False) as pbar:
                for itr, (img, msg) in enumerate(pbar):
                    loss_dict, img_dict = trainer.test(self.model, img, msg)
                    meter.updates(loss_dict())

                    pbar.set_postfix_str(f'loss={loss_dict["total"]:.4f}')

        return meter.to_dict, img_dict
