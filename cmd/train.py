import os
import sys

import distortion
import dotenv
import hydra
import omegaconf
import torch
import torch.nn as nn

sys.path.append(os.path.abspath("."))

from pkg.experiment import (
    Experiment,
    ExperimentConfig,
    CometConfig,
)
from pkg.data_controller import (
    DataController,
)
from pkg.model import (
    HiddenModel,
)
from pkg.iterator import (
    TrainConfig,
    train_iter,
)
from pkg.cycle import (
    HiddenCycle,
    HiddenLossConfig,
    HiddenTrainConfig,
)


dotenv.load_dotenv()

@hydra.main(config_name="config/train.yaml")
def main(cfg):
    is_config_valid(cfg)

    expcfg = ExperimentConfig(
        name=cfg.experiment.name,
        tags=cfg.experiment.tags,
        use_comet=cfg.experiment.use_comet,
        resume_training=cfg.experiment.resume_training,
        comet=CometConfig(
            project=os.environ["COMET_PROJECT"],
            workspace=os.environ["COMET_WORKSPACE"],
            api_key=os.environ["COMET_API_KEY"],
            resume_exp_key=cfg.experiment.resume_exp_key,
        )
    )
    experiment = Experiment(expcfg)
    experiment.log_experiment_params(omegaconf.OmegaConf.to_container(cfg))

    device = torch.device("cpu")

    datacon = DataController(
        cfg.data.train_path, cfg.data.test_path,
        cfg.data.batch_size, cfg.data.msg_len, cfg.data.resol,
    )

    last_epoch = 0
    ckpt = None
    if cfg.experiment.resume_training:
        start_epoch, ckpt = experiment.load_ckpt()

    distortioner = distortion.Identity()
    model = HiddenModel(distortioner)

    train_cycle = HiddenCycle(HiddenLossConfig(), model, device, cfg.gpu_ids)
    train_cycle.setup_train(HiddenTrainConfig(), ckpt)

    train_iter_cfg = TrainConfig(
        epochs=cfg.training.epochs,
        start_epoch=last_epoch+1,
        test_interval=cfg.training.test_interval,
    )
    train_iter(train_iter_cfg, train_cycle, datacon, experiment)

    experiment.save_parameters(train_cycle.get_parameters())


def is_config_valid(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
