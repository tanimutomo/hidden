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
import pkg.dataset
import pkg.distorter
import pkg.experiment
import pkg.iterator
import pkg.model
import pkg.seed


dotenv.load_dotenv()


@hydra.main(config_name="config/train.yaml")
def main(cfg):
    validate_config(cfg)
    if cfg.seed: pkg.seed.set_seed(cfg.seed)

    if cfg.experiment.name == "debug":
        pkg.experiment.debug_init(epochs=cfg.training.epochs)
    elif cfg.experiment.name == "new":
        pkg.experiment.new_init(
            cfg=pkg.experiment.NewConfig(
                name=cfg.experiment.name,
                tags=cfg.experiment.tags,
                use_comet=cfg.experiment.use_comet,
                comet=pkg.experiment.CometConfig(
                    project=os.environ["COMET_PROJECT"],
                    workspace=os.environ["COMET_WORKSPACE"],
                    api_key=os.environ["COMET_API_KEY"],
                ),
            ),
        )
    elif cfg.experiment.name == "resume":
        pkg.experiment.resume_init(
            cfg=pkg.experiment.ResumeConfig(
                name=cfg.experiment.name,
                use_comet=cfg.experiment.use_comet,
                comet_key=cfg.experiment.comet_key,
                comet=pkg.experiment.CometConfig(
                    project=os.environ["COMET_PROJECT"],
                    workspace=os.environ["COMET_WORKSPACE"],
                    api_key=os.environ["COMET_API_KEY"],
                )
            )
        )
    else:
        raise ValueError()
    pkg.experiment.log_hyper_parameters(omegaconf.OmegaConf.to_container(cfg))

    device = torch.device(f"cuda:{cfg.gpu_ids[0]}" if cfg.gpu_ids else "cpu")

    datastats = pkg.dataset.COCODatasetStats()
    if cfg.dataset.name == "bit":
        w2v = None
        train_dataset = pkg.dataset.BitMessageDataset(
            root_dir=cfg.data.train_path,
            msg_len=cfg.dataset.msg_len,
        )
        test_dataset = pkg.dataset.BitMessageDataset(
            root_dir=cfg.data.test_path,
            msg_len=cfg.dataset.msg_len,
        )
    elif cfg.dataset.name == "word":
        w2v = pkg.wordvec.GloVe(use_words=cfg.dataset.use_words, num_words=cfg.dataset.num_words, dim=cfg.dataset.dim)
        train_dataset = pkg.dataset.WordMessageDataset(
            root_dir=cfg.data.train_path,
            word_vec=w2v,
        )
        test_dataset = pkg.dataset.WordMessageDataset(
            root_dir=cfg.data.test_path,
            word_vec=w2v,
        )
    else:
        raise NotImplementedError()
    datactl = pkg.data_controller.DataController(
        resol=cfg.data.resol,
        dataset_stats=datastats,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_batch_size=cfg.data.train_batch_size,
        test_batch_size=cfg.data.test_batch_size,
    )

    last_epoch = 0
    ckpt = None
    if cfg.experiment.resume_training:
        last_epoch, ckpt = pkg.experiment.get_checkpoint()

    pkg.distorter.init(datastats.means(), datastats.stds())
    model = pkg.model.HiddenModel(
        msg_len=cfg.dataset.msg_len if cfg.dataset.name == "bit" else cfg.dataset.dim * cfg.dataset.num_words,
        train_distorter=pkg.distorter.get(pkg.distorter.Config(
            name=cfg.train_distortion.name,
            p=cfg.train_distortion.probability,
            w=cfg.train_distortion.kernel_size,
            s=cfg.train_distortion.sigma,
            qf=cfg.train_distortion.quality_factor,
            ps=cfg.train_distortion.probabilities,
            ss=cfg.train_distortion.sigmas,
        )),
        test_distorter=pkg.distorter.get(pkg.distorter.Config(
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

    if cfg.dataset.name == "bit":
        train_cycle = pkg.cycle.HiddenCycle(
            loss_cfg=pkg.cycle.HiddenLossConfig(),
            model=model,
            device=device,
            gpu_ids=cfg.gpu_ids,
        )
    elif cfg.dataset.name == "word":
        train_cycle = pkg.cycle.WordHiddenCycle(
            loss_cfg=pkg.cycle.HiddenLossConfig(),
            model=model,
            w2v=w2v,
            device=device,
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
        log=pkg.iterator.log_bit_outputs if cfg.dataset.name == "bit" else pkg.iterator.log_word_outputs,
        metrics=pkg.cycle.HiddenMetricOutput.keys() if cfg.dataset.name == "bit" else pkg.cycle.WordHiddenMetricOutput.keys(),
    )
    print("End Training")

    pkg.experiment.log_parameters(train_cycle.get_parameters())


def validate_config(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
