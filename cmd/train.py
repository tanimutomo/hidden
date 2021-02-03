from comet_ml import Experiment

import os
import sys

import distortion
import hydra
import omegaconf
import torch
import torch.nn as nn

sys.path.append(os.path.abspath("./pkg"))

from data import (
    WatermarkDataset
)
from transform import (
    ImageTransformer,
)
from model import (
    HiddenModel,
)
from loss import (
    HiddenLoss,
)
from trainer import (
    Trainer,
    TrainerConfig,
)


@hydra.main(config_name="config/train.yaml")
def main(cfg):
    is_config_valid(cfg)
    # if cfg.seed is not None:
    #     set_seed(cfg.seed)

    # # Experiment and Set comet
    # experiment = ExperimentController(cfg)
    # if cfg.comet.use:
    #     experiment.set_comet(os.path.join(get_original_cwd(), '.comet'))

    # Device
    # device = torch.device(f'cuda:{cfg.gpu_ids[0]}')
    device = torch.device('cpu')

    transformer = ImageTransformer(cfg.data.resol)

    train_loader = torch.utils.data.DataLoader(
        WatermarkDataset(cfg.data.train_path, cfg.data.msg_len, transformer.train),
        cfg.data.batch_size,
        True,
    )
    test_loader = torch.utils.data.DataLoader(
        WatermarkDataset(cfg.data.test_path, cfg.data.msg_len, transformer.test),
        cfg.data.batch_size,
        False,
    )

    # resume training
    # last_iter, model_sd, optimizer_sd = experiment.load_ckpt()

    # model
    distortioner = distortion.Identity()
    model = HiddenModel(cfg.model, distortioner)
    # model = get_model(
    #     cfg.model,
    #     FourierTransform(cutidx=cfg.loss.spec.cut_idx),
    #     InverseFourierTransform(cutidx=cfg.loss.spec.cut_idx),
    # )
    # if model_sd is not None:
    #     model.load_state_dict(model_sd)
    model.to(device)
    if len(cfg.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)

    # optimizer
    if cfg.optim.method == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )
    else:
        raise NotImplementedError(f"{cfg.optim.method} is not implemented.")
    # if optimizer_sd:
    #     optimizer.load_state_dict(optimizer_sd)

    # Loss fucntion
    # criterion = get_loss(cfg.loss).to(device)
    loss = HiddenLoss()
    loss.to(device)

    trainer = Trainer(device, model, transformer, None)
    trainer.train_setup(TrainerConfig(cfg.trainer.save_img_interval, cfg.trainer.test_interval))
    trainer.train(train_loader, test_loader, loss, optimizer)

    # experiment.save_model(trainer.model)


def is_config_valid(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
