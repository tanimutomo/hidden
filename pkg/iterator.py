from dataclasses import dataclass, field
import os
import sys
import typing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

import pkg.meter
import pkg.cycle
import pkg.data_controller
import pkg.experiment


@dataclass
class TrainConfig:
    epochs: int =100
    start_epoch: int =1
    test_interval: int =10
    lr_scheduler_milestones: typing.List[int] =field(default_factory=list)
    lr_scheduler_step_factor: float =1.0
    lr_scheduler_state_dict: typing.Dict =field(default_factory=dict)


def train_iter(
    cfg: TrainConfig,
    trainer: pkg.cycle.Cycle,
    datactl: pkg.data_controller.DataController,
    experiment: pkg.experiment.Experiment,
):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        trainer.optimizer,
        milestones=cfg.lr_scheduler_milestones,
        gamma=cfg.lr_scheduler_step_factor,
        last_epoch=cfg.start_epoch-2, # scheduler expect to last_epoch=-1 for a new training.
    )
    if cfg.lr_scheduler_state_dict:
        scheduler.load_state_dict(cfg.lr_scheduler_state_dict)

    for epoch in range(cfg.start_epoch, cfg.epochs+1):
        meter = pkg.meter.MultiAverageMeter(trainer.metric_keys)

        trainer.model.train()
        with tqdm(datactl.train_loader, ncols=80, leave=False) as pbar:
            for step, item in enumerate(pbar):
                metric_dict, img_dict = trainer.train(item)
                meter.updates(metric_dict)

                pbar.set_postfix_str(f'metric={metric_dict[trainer.metric_keys[-1]]:.4f}')

        experiment.epoch_report(meter.to_dict(), "train", epoch, cfg.epochs)
        experiment.save_image(img_dict, "train", epoch, datactl.img_post_transformer)

        if epoch % cfg.test_interval == 0:
            test_iter(trainer, datactl, experiment, epoch, cfg.epochs)

        ckpt = {
            "scheduler": scheduler.state_dict(),
            "trainer": trainer.get_checkpoint(),
        }
        experiment.save_checkpoint(ckpt, epoch)


def test_iter(
    tester: pkg.cycle.Cycle,
    datactl: pkg.data_controller.DataController,
    experiment: pkg.experiment.Experiment,
    epoch: int =0,
    epochs: int =0
):
    meter = pkg.meter.MultiAverageMeter(tester.metric_keys)
    tester.model.eval()

    with torch.no_grad():
        with tqdm(datactl.test_loader, ncols=80, leave=False) as pbar:
            for step, item in enumerate(pbar):
                metric_dict, img_dict = tester.test(item)
                meter.updates(metric_dict)

                pbar.set_postfix_str(f'metric={metric_dict[tester.metric_keys[-1]]:.4f}')

    experiment.epoch_report(metric_dict, "test", epoch, epochs)
    experiment.save_image(img_dict, "test", epoch, datactl.img_post_transformer)
