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

    datactl = pkg.data_controller.DataController(
        msg_len=cfg.data.msg_len,
        resol=cfg.data.resol,
        test_dataset_path=cfg.data.test_path,
        test_batch_size=cfg.data.test_batch_size,
        require_trainset=False,
    )

    params = experiment.load_parameters(cfg.experiment.relative_model_path)

    distortioner = pkg.distortion.get(pkg.distortion.Config(
        name=cfg.distortion.name,
        p=cfg.distortion.probability,
        w=cfg.distortion.kernel_size,
        s=cfg.distortion.sigma,
        qf=cfg.distortion.quality_factor,
    ))
    model = pkg.model.HiddenModel(distortioner=distortioner)

    test_cycle = pkg.cycle.HiddenCycle(
        loss_cfg=pkg.cycle.HiddenLossConfig(),
        model=model, device=device, gpu_ids=cfg.gpu_ids
    )
    test_cycle.setup_test(cfg=pkg.cycle.HiddenTestConfig, params=params)

    pkg.iterator.test_iter(
        tester=test_cycle, datactl=datactl,
        experiment=experiment,
    )


def validate_config(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
