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
from pkg.trainer import (
    Trainer,
)
from pkg.experiment import (
    Experiment,
)


class TrainIteratorConfig(typing.NamedTuple):
    epochs: int =100
    test_interval: int =10


class TrainIterator(object):
    def __init__(self, cfg: TrainIteratorConfig, experiment: Experiment):
        self.cfg = cfg
        self.experiment: Experiment = experiment

    def train(self, train_loader: DataLoader, test_loader: DataLoader, trainer: Trainer):
        print("Start Training...")

        for epoch in range(self.cfg.epochs):
            self.meter = MultiAverageMeter(trainer.loss_keys)

            trainer.model.train()
            for step, (img, msg) in enumerate(train_loader):
                loss_dict, img_dict = trainer.train(img, msg)
                self.meter.updates(loss_dict)
                print(f"step: {step}\n\tloss: {loss_dict}")

            self.experiment.epoch_report(self.meter.to_dict, "train", epoch, self.cfg.epochs)
            self.experiment.save_image(img_dict, epoch)

            if step % self.cfg.test_interval == 0:
                loss_dict, img_dict = self.test(test_loader, trainer)
                self.experiment.epoch_report(loss_dict, "test", epoch, self.cfg.epochs)
                self.experiment.save_image(img_dict, epoch)

            self.experiment.save_ckpt(trainer.get_checkpoint(), epoch)

    def test(self, test_loader: DataLoader, trainer: Trainer) -> typing.Tuple[dict, dict]:
        meter = MultiAverageMeter(trainer.loss_keys)
        trainer.model.eval()

        with torch.no_grad():
            with tqdm(test_loader, ncols=80, leave=False) as pbar:
                for itr, (img, msg) in enumerate(pbar):
                    loss_dict, img_dict = trainer.test(img, msg)
                    meter.updates(loss_dict())

                    pbar.set_postfix_str(f'loss={loss_dict["total"]:.4f}')

        return meter.to_dict, img_dict

