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
    test_iter,
)
from pkg.cycle import (
    HiddenCycle,
    HiddenLossConfig,
    HiddenTestConfig,
)
from pkg.seed import (
    set_seed,
)


dotenv.load_dotenv()

@hydra.main(config_name="config/test.yaml")
def main(cfg):
    validate_config(cfg)
    if cfg.seed: set_seed(cfg.seed)

    expcfg = ExperimentConfig(
        name=cfg.experiment.name,
        tags=cfg.experiment.tags,
        use_comet=cfg.experiment.use_comet,
        comet=CometConfig(
            project=os.environ["COMET_PROJECT"],
            workspace=os.environ["COMET_WORKSPACE"],
            api_key=os.environ["COMET_API_KEY"],
        )
    )
    experiment = Experiment(expcfg)
    experiment.log_experiment_params(omegaconf.OmegaConf.to_container(cfg))

    device = torch.device(f"cuda:{cfg.gpu_ids[0]}" if cfg.gpu_ids else "cpu")

    datacon = DataController(
        cfg.data.train_path, cfg.data.test_path,
        cfg.data.batch_size, cfg.data.msg_len, cfg.data.resol,
    )

    params = experiment.load_parameters(cfg.experiment.relative_model_path)

    distortioner = distortion.Identity()
    model = HiddenModel(distortioner)

    test_cycle = HiddenCycle(HiddenLossConfig(), model, device, cfg.gpu_ids)
    test_cycle.setup_test(HiddenTestConfig, params)

    test_iter(test_cycle, datacon, experiment)


def validate_config(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
