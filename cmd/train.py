import os
import sys

import distortion
import hydra
import omegaconf
import torch
import torch.nn as nn

sys.path.append(os.path.abspath("./pkg"))

from experiment import (
    Experiment,
)
from data import (
    WatermarkDataset
)
from transform import (
    ImageTransformer,
)
from model import (
    HiddenModel,
)
from train_iterator import (
    TrainIteratorConfig,
    TrainIterator,
)
from trainer import (
    HiddenTrainer,
)


@hydra.main(config_name="config/train.yaml")
def main(cfg):
    is_config_valid(cfg)
    # if cfg.seed is not None:
    #     set_seed(cfg.seed)

    experiment = Experiment(cfg)
    experiment.set_comet(os.path.join(hydra.utils.get_original_cwd(), '.comet'))

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
    trainer = HiddenTrainer(device, cfg.gpu_ids, model)

    cfg = TrainIteratorConfig(cfg.trainer.save_img_interval, cfg.trainer.test_interval)
    iterator = TrainIterator(cfg, model, None)
    iterator.train(train_loader, test_loader, trainer)

    experiment.save_model(trainer.model)


def is_config_valid(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
