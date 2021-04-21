import os
import sys

import comet_ml
import dotenv
import hydra
import omegaconf
import torch

sys.path.append(os.path.abspath("."))

import pkg.cycle
import pkg.data_controller
import pkg.distortion
import pkg.experiment
import pkg.iterator
import pkg.model
import pkg.seed


dotenv.load_dotenv()


@hydra.main(config_name="config/train.yaml")
def main(cfg):
    validate_config(cfg)
    if cfg.seed: pkg.seed.set_seed(cfg.seed)

    experiment = pkg.experiment.Experiment(
        cfg=pkg.experiment.ExperimentConfig(
            name=cfg.experiment.name,
            tags=cfg.experiment.tags,
            use_comet=cfg.experiment.use_comet,
            resume_training=cfg.experiment.resume_training,
            comet=pkg.experiment.CometConfig(
                project=os.environ["COMET_PROJECT"],
                workspace=os.environ["COMET_WORKSPACE"],
                api_key=os.environ["COMET_API_KEY"],
                resume_experiment_key=cfg.experiment.comet.resume_experiment_key,
            )
        ),
    )
    experiment.log_experiment_params(omegaconf.OmegaConf.to_container(cfg))

    datastats = pkg.dataset.COCODatasetStats()
    datactl = pkg.data_controller.DataController(
        msg_len=cfg.data.msg_len,
        resol=cfg.data.resol,
        dataset_stats=datastats,
        train_dataset_path=cfg.data.train_path,
        test_dataset_path=cfg.data.test_path,
        train_batch_size=cfg.data.train_batch_size,
        test_batch_size=cfg.data.test_batch_size,
    )

    last_epoch = 0
    ckpt = None
    if cfg.experiment.resume_training:
        last_epoch, ckpt = experiment.load_checkpoint()

    pkg.distortion.init(datastats.means(), datastats.stds())
    model = pkg.model.HiddenModel(
        train_distortioner=pkg.distortion.get(pkg.distortion.Config(
            name=cfg.train_distortion.name,
            p=cfg.train_distortion.probability,
            w=cfg.train_distortion.kernel_size,
            s=cfg.train_distortion.sigma,
            qf=cfg.train_distortion.quality_factor,
            ps=cfg.train_distortion.probabilities,
            ss=cfg.train_distortion.sigmas,
        )),
        test_distortioner=pkg.distortion.get(pkg.distortion.Config(
            name=cfg.test_distortion.name,
            p=cfg.test_distortion.probability,
            w=cfg.test_distortion.kernel_size,
            s=cfg.test_distortion.sigma,
            qf=cfg.test_distortion.quality_factor,
            ps=cfg.test_distortion.probabilities,
            ss=cfg.test_distortion.sigmas,
        )),
        train_distortion_parallelable=cfg.train_distortion.parallelable,
        test_distortion_parallelable=cfg.test_distortion.parallelable,
    )

    train_cycle = pkg.cycle.HiddenCycle(
        loss_cfg=pkg.cycle.HiddenLossConfig(),
        model=model,
        device=torch.device(f"cuda:{cfg.gpu_ids[0]}" if cfg.gpu_ids else "cpu"),
        gpu_ids=cfg.gpu_ids,
    )
    train_cycle.setup_train(
        cfg=pkg.cycle.HiddenTrainConfig(
            optimizer_lr=cfg.train.optimizer_lr,
            optimizer_wd=cfg.train.optimizer_wd,
            discriminator_lr=cfg.train.discriminator_lr,
        ),
        ckpt=ckpt["trainer"] if ckpt else None,
    )

    print("Start Training...")
    pkg.iterator.train_iter(
        cfg=pkg.iterator.TrainConfig(
            epochs=cfg.training.epochs,
            start_epoch=last_epoch+1,
            test_interval=cfg.training.test_interval,
            lr_scheduler_milestones=cfg.training.lr_scheduler_milestones,
            lr_scheduler_step_factor=cfg.training.lr_scheduler_step_factor,
            lr_scheduler_state_dict=ckpt["scheduler"] if ckpt else None,
        ),
        trainer=train_cycle, 
        datactl=datactl, 
        experiment=experiment,
    )
    print("End Training")

    experiment.save_parameters(train_cycle.get_parameters(), cfg.training.epochs)


def validate_config(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
