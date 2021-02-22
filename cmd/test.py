import os
import sys

import comet_ml
import distortion
import dotenv
import hydra
import omegaconf
import torch

sys.path.append(os.path.abspath("."))

import pkg.experiment
import pkg.data_controller
import pkg.model
import pkg.iterator
import pkg.cycle
import pkg.seed


dotenv.load_dotenv()


@hydra.main(config_name="config/test.yaml")
def main(cfg):
    validate_config(cfg)
    if cfg.seed: pkg.seed.set_seed(cfg.seed)

    expcfg = pkg.experiment.ExperimentConfig(
        name=cfg.experiment.name,
        tags=cfg.experiment.tags,
        use_comet=cfg.experiment.use_comet,
        comet=pkg.experiment.CometConfig(
            project=os.environ["COMET_PROJECT"],
            workspace=os.environ["COMET_WORKSPACE"],
            api_key=os.environ["COMET_API_KEY"],
        )
    )
    experiment = pkg.experiment.Experiment(expcfg)
    experiment.log_experiment_params(omegaconf.OmegaConf.to_container(cfg))

    device = torch.device(f"cuda:{cfg.gpu_ids[0]}" if cfg.gpu_ids else "cpu")

    datacon = pkg.data_controller.DataController(
        cfg.data.train_path, cfg.data.test_path,
        cfg.data.batch_size, cfg.data.msg_len, cfg.data.resol,
    )

    params = experiment.load_parameters(cfg.experiment.relative_model_path)

    distortioner = distortion.Identity()
    model = pkg.model.HiddenModel(distortioner)

    test_cycle = pkg.cycle.HiddenCycle(pkg.cycle.HiddenLossConfig(), model, device, cfg.gpu_ids)
    test_cycle.setup_test(pkg.cycle.HiddenTestConfig, params)

    pkg.iterator.test_iter(test_cycle, datacon, experiment)


def validate_config(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
