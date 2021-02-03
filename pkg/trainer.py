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


class TrainerConfig(typing.NamedTuple):
    save_img_interval: int =10
    test_interval: int =10


class HiddenTrainer(object):
    def __init__(self, device, model, transformer, experiment):
        self.device = device
        self.model = model
        self.transformer = transformer
        # self.experiment = experiment

    def train_setup(self, cfg: TrainerConfig):
        self.cfg = cfg

    def train(self, train_loader, test_loader, loss: Loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.meter = MultiAverageMeter(self.loss.names)

        print("Start Training...")
        for step, (img, msg) in enumerate(train_loader):
            # self.experiment.iter = itr
            loss_dict, save_imgs = self._train_one(img, msg)
            self.meter.updates(loss_dict)
            # self.experiment.report("train", loss_dict)

            if step % self.cfg.img_interval == 0:
                pass
                # self.experiment.save_image(save_inp, f"train_{itr}.png")
            
            if step % self.cfg.test_interval == 0:
                loss_dict, save_imgs = self.test(test_loader, self.loss)
                # self.experiment.report("test", loss_dict)
                # self.experiment.save_image(save_inp, f"test_{itr}.png")

            #     self.experiment.save_ckpt(self.model, self.optimizer)

    def test(self, test_loader, loss: Loss) -> typing.Tuple[dict, dict]:
        meter = MultiAverageMeter(loss.names)
        self.model.eval()

        with torch.no_grad():
            with tqdm(test_loader, ncols=80, leave=False) as pbar:
                for itr, (gt, mask) in enumerate(pbar):
                    img = img.to(self.device)
                    msg = msg.to(self.device)

                    enc_img, pred_msg = self.model(img, msg)
                    loss.calcurate(enc_img, pred_msg, img, msg)
                    meter.updates(loss.to_item_dict())

                    pbar.set_postfix_str(f'loss={loss.losses["total"].item():.4f}')

        return (
            loss.to_item_dict(),
            torch.stack([enc_img[0], img[0]], dim=0).cpu().detach(),
        )

    def _train_one(self, img, msg) -> typing.Tuple[dict, dict]:
        self.model.train()

        img = img.to(self.device)
        msg = msg.to(self.device)

        enc_img, pred_msg = self.model(img, msg)
        self.loss.calcurate(enc_img, pred_msg, img, msg)

        self.loss.discriminator_optimize()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


        return self.loss.to_item_dict, torch.stack([enc_img[0], img[0]]).cpu().detach()
