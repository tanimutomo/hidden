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
from pkg.data_controller import (
    DataController,
)
from pkg.experiment import (
    Experiment,
)


class TrainIteratorConfig(typing.NamedTuple):
    epochs: int =100
    start_epoch: int =0
    test_interval: int =10


@dataclass
class TrainIterator(object):
    cfg: TrainIteratorConfig
    experiment: Experiment
    datacon: DataController

    def train(self, trainer: Trainer):
        print("Start Training...")

        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            self.meter = MultiAverageMeter(trainer.loss_keys)

            trainer.model.train()
            with tqdm(self.datacon.train_loader, ncols=80, leave=False) as pbar:
                for step, (img, msg) in enumerate(pbar):
                    loss_dict, img_dict = trainer.train(img, msg)
                    self.meter.updates(loss_dict)

                    pbar.set_postfix_str(f'loss={loss_dict[trainer.loss_keys[-1]]:.4f}')
                    if step == 1: break # DEBUG

            self.experiment.epoch_report(self.meter.to_dict(), "train", epoch, self.cfg.epochs)
            self.experiment.save_image(img_dict, epoch, self.datacon.img_post_transformer)

            if step % self.cfg.test_interval == 0:
                loss_dict, img_dict = self.test(trainer, epoch)

            self.experiment.save_ckpt(trainer.get_checkpoint(), epoch)

    def test(self, trainer: Trainer, epoch: int =0) -> typing.Tuple[dict, dict]: #TODO: Trainerではなく，Testerにする
        meter = MultiAverageMeter(trainer.loss_keys)
        trainer.model.eval()

        with torch.no_grad():
            with tqdm(self.datacon.test_loader, ncols=80, leave=False) as pbar:
                for step, (img, msg) in enumerate(pbar):
                    loss_dict, img_dict = trainer.test(img, msg)
                    meter.updates(loss_dict)

                    pbar.set_postfix_str(f'loss={loss_dict[trainer.loss_keys[-1]]:.4f}')
                    if step == 1: break # DEBUG

        self.experiment.epoch_report(loss_dict, "test", epoch, self.cfg.epochs)
        self.experiment.save_image(img_dict, epoch, self.datacon.img_post_transformer)
        return meter.to_dict(), img_dict

