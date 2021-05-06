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
    log: typing.Callable,
    metrics: list,
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
        meter = pkg.meter.MultiAverageMeter(metrics)
        trainer.model.train()
        pkg.experiment.set_epoch(epoch)
        pkg.experiment.set_mode("train")

        with tqdm(datactl.train_loader, ncols=80, leave=False) as pbar:
            for step, item in enumerate(pbar):
                output = trainer.train(item)
                meter.updates(output.metric.todict())

                pbar.set_postfix_str(f"metric={output.metric.message_accuracy:.4f}")

        log(output)
        if epoch % cfg.test_interval == 0:
            test_iter(trainer, datactl, log, metrics)

        ckpt = {
            "scheduler": scheduler.state_dict(),
            "trainer": trainer.get_checkpoint(),
        }
        pkg.experiment.log_checkpoint(ckpt)


def test_iter(
    tester: pkg.cycle.Cycle,
    datactl: pkg.data_controller.DataController,
    log: typing.Callable,
    metrics: list,
):
    meter = pkg.meter.MultiAverageMeter(metrics)
    tester.model.eval()
    pkg.experiment.set_mode("test")

    with torch.no_grad():
        with tqdm(datactl.test_loader, ncols=80, leave=False) as pbar:
            for step, item in enumerate(pbar):
                output = tester.test(item)
                meter.updates(output.metric.todict())

                pbar.set_postfix_str(f"metric={output.metric.message_accuracy:.4f}")

    log(output)


def log_bit_outputs(output: pkg.cycle.HiddenOutput):
    pkg.experiment.log_metrics(output.metric.todict())
    pkg.experiment.log_images(output.image.todict())


def log_word_outputs(output: pkg.cycle.WordHiddenOutput):
    pkg.experiment.log_metrics(output.metric.todict())
    pkg.experiment.log_images(output.image.todict())
    pkg.experiment.log_texts(output.text.todict())
