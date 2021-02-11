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
from pkg.cycle import (
    Cycle,
)
from pkg.data_controller import (
    DataController,
)
from pkg.experiment import (
    Experiment,
)


@dataclass
class TrainConfig:
    epochs: int =100
    start_epoch: int =0
    test_interval: int =10


def train_iter(cfg: TrainConfig, trainer: Cycle, datacon: DataController, experiment: Experiment):
    print("Start Training...")

    for epoch in range(cfg.start_epoch, cfg.epochs):
        meter = MultiAverageMeter(trainer.loss_keys)

        trainer.model.train()
        with tqdm(datacon.train_loader, ncols=80, leave=False) as pbar:
            for step, (img, msg) in enumerate(pbar):
                loss_dict, img_dict = trainer.train(img, msg)
                meter.updates(loss_dict)

                pbar.set_postfix_str(f'loss={loss_dict[trainer.loss_keys[-1]]:.4f}')
                if step == 1: break # DEBUG

        experiment.epoch_report(meter.to_dict(), "train", epoch, cfg.epochs)
        experiment.save_image(img_dict, epoch, datacon.img_post_transformer)

        if step % cfg.test_interval == 0:
            test_iter(trainer, datacon, experiment, epoch, cfg.epochs)

        experiment.save_checkpoint(trainer.get_checkpoint(), epoch)


def test_iter(tester: Cycle, datacon: DataController, experiment: Experiment, epoch: int =0, epochs: int =0):
    meter = MultiAverageMeter(tester.loss_keys)
    tester.model.eval()

    with torch.no_grad():
        with tqdm(datacon.test_loader, ncols=80, leave=False) as pbar:
            for step, (img, msg) in enumerate(pbar):
                loss_dict, img_dict = tester.test(img, msg)
                meter.updates(loss_dict)

                pbar.set_postfix_str(f'loss={loss_dict[tester.loss_keys[-1]]:.4f}')
                if step == 1: break # DEBUG

    experiment.epoch_report(loss_dict, "test", epoch, epochs)
    experiment.save_image(img_dict, epoch, datacon.img_post_transformer)
