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
from pkg.data import (
    WatermarkDataset
)
from pkg.transform import (
    ImageTransformer,
)
from pkg.model import (
    HiddenModel,
)
from pkg.train_iterator import (
    TrainIteratorConfig,
    TrainIterator,
)
from pkg.trainer import (
    HiddenTrainer,
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

    transformer = ImageTransformer(cfg.data.resol)

    train_loader = torch.utils.data.DataLoader(
        WatermarkDataset(cfg.data.train_path, cfg.data.msg_len, transformer.train),
        cfg.data.batch_size, True,
    )
    test_loader = torch.utils.data.DataLoader(
        WatermarkDataset(cfg.data.test_path, cfg.data.msg_len, transformer.test),
        cfg.data.batch_size, False,
    )

    start_epoch = 0
    ckpt = None
    if cfg.experiment.resume_training:
        start_epoch, ckpt = experiment.load_ckpt()

    distortioner = distortion.Identity()
    model = HiddenModel(distortioner)
    trainer = HiddenTrainer(device=device, gpu_ids=cfg.gpu_ids, model=model, ckpt=ckpt)

    trncfg = TrainIteratorConfig(
        epochs=cfg.training.epochs,
        start_epoch=start_epoch,
        test_interval=cfg.training.test_interval,
    )
    iterator = TrainIterator(trncfg, experiment)
    iterator.train(train_loader, test_loader, trainer)

    experiment.save_model(trainer.model_state_dict())


def is_config_valid(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
