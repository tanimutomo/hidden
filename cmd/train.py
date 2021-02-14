import os
import sys

import comet_ml
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
from pkg.seed import (
    set_seed,
)


dotenv.load_dotenv()

@hydra.main(config_name="config/train.yaml")
def main(cfg):
    validate_config(cfg)
    if cfg.seed: set_seed(cfg.seed)

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

    device = torch.device(f"cuda:{cfg.gpu_ids[0]}" if cfg.gpu_ids else "cpu")

    datacon = DataController(
        cfg.data.train_path, cfg.data.test_path,
        cfg.data.batch_size, cfg.data.msg_len, cfg.data.resol,
    )

    last_epoch = 0
    ckpt = None
    if cfg.experiment.resume_training:
        last_epoch, ckpt = experiment.load_checkpoint()

    distortioner = distortion.Identity()
    model = HiddenModel(distortioner)

    train_cycle = HiddenCycle(HiddenLossConfig(), model, device, cfg.gpu_ids)
    train_cycle.setup_train(
        HiddenTrainConfig(
            cfg.train.optimizer_lr,
            cfg.train.optimizer_wd,
            cfg.train.discriminator_lr,
        ),
        ckpt,
    )

    train_iter_cfg = TrainConfig(
        epochs=cfg.training.epochs,
        start_epoch=last_epoch+1,
        test_interval=cfg.training.test_interval,
    )
    print("Start Training...")
    train_iter(train_iter_cfg, train_cycle, datacon, experiment)
    print("End Training")

    experiment.save_parameters(train_cycle.get_parameters(), cfg.training.epochs)


def validate_config(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
