import comet_ml
import csv
import os
import shutil
import sys

import moment
import torch
import torchvision
import yaml
import typing

sys.path.append(os.path.abspath("."))

from pkg.meter import (
    AverageMeter
)

Metrics = typing.Dict[str, float]
Images = typing.Dict[str, torch.FloatTensor]

CHECKPOINT_PATH = "checkpoint.pth"
MODEL_PATH = "model.pth"


class CometConfig(typing.NamedTuple):
    project: str
    workspace: str
    api_key: str
    resume_exp_key: str


class ExperimentConfig(typing.NamedTuple):
    name: str
    tags: typing.Dict[str, str]
    comet: CometConfig
    use_comet: bool =False
    resume_training: bool =False


class Experiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.comet: comet_ml.BaseExperiment = None

        if cfg.resume_training:
            if not os.path.exists(self.cfg.name):
                raise ValueError(f"The specified experiment ({self.cfg.name}) is not existed.")
            if cfg.use_comet and not cfg.comet.resume_exp_key:
                raise ValueError(f"cfg.comet.resume_exp_key is empty.")

        if cfg.use_comet:
            self._set_comet()

        print(f"Start experiment: {self.cfg.name}")

    def _set_comet(self):
        exp_args = dict(
            project_name=self.cfg.comet.project,
            workspace=self.cfg.comet.workspace,
            api_key=self.cfg.comet.api_key,
            auto_param_logging=False,
            auto_metric_logging=False,
            log_env_host=True,
            parse_args=False
        )
        if self.cfg.comet.resume_exp_key:
            exp_args["previous_experiment"] = self.cfg.comet.resume_key
            comet = comet_ml.ExistingExperiment(**exp_args)
        else:
            comet = comet_ml.Experiment(**exp_args)
            comet.set_name(self.cfg.name)

        if self.self.cfg.tags:
            comet.add_tags(self.self.cfg.tags)

        self.comet = comet

    def log_experiment_params(self, params: dict):
        if self.cfg.use_comet:
            self.comet.log_parameters(_to_flat_dict(params, dict()))

    def epoch_report(self, metrics: Metrics, mode: str, epoch: int, epochs: int, test=False):
        stdout = f"{mode.upper()} [{epoch:d}/{epochs:d}]  "

        for name, value in metrics.items():
            stdout += f"{name} = {value:.4f} / "
            if not self.comet: continue
            self.comet.log_metric( f"{mode}-{name}", value, step=epoch)

        self._save_metrics(metrics, mode, epoch)
        print(stdout)

    def _save_metrics(self, _metrics: Metrics, mode: str, epoch: int):
        metrics = dict()
        for name, value in _metrics.items():
            metrics[f"{mode}-{name}"] = value
        metrics.update({"epoch": epoch, "timestamp": moment.now().format("YYYY-MMDD-HHmm-ss")})

        path = _get_metrics_path(mode)
        open_mode = "a" if os.path.exists(path) else "w"
        with open(path, open_mode) as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if open_mode == "w":
                writer.writeheader()
            writer.writerow(metrics)

        if self.comet:
            self._send_file_to_comet(path, epoch, overwrite=True)

    def _send_file_to_comet(self, path :str, epoch: int, overwrite=False):
        self.comet.log_asset(path, overwrite=overwrite, step=epoch)

    def save_ckpt(self, ckpt: dict, epoch: int):
        ckpt["epoch"] = epoch
        torch.save(ckpt, CHECKPOINT_PATH)
        if self.comet:
            self._send_file_to_comet(CHECKPOINT_PATH, epoch, overwrite=True)

    def load_ckpt(self) -> typing.Tuple[int, dict]:
        if not self.cfg.resume_training:
            raise ValueError("This training is new experiment.")
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        shutil.copy(CHECKPOINT_PATH, CHECKPOINT_PATH+f".bkup.{moment.now().format('YYYY-MMDD-HHmm-ss')}")
        epoch = ckpt.pop("epoch")
        return epoch, ckpt

    def save_model(self, model_sd: dict):
        torch.save(model_sd, MODEL_PATH)
        if self.comet:
            self.send_file(MODEL_PATH)

    def save_image(self, imgs: Images, epoch: int):
        os.makedirs("images", exist_ok=True)
        for name, img in imgs.items():
            filename = f"{name}_{epoch}.png"
            path = os.path.join("images", filename)
            torchvision.utils.save_image(img, path)
            self._send_image_to_comet(path, epoch)
    
    def _send_image_to_comet(self, path :str, epoch: int):
        if self.comet:
            self.comet.log_image(path, step=epoch)


def _to_flat_dict(target :dict, fdict: dict):
    if len(list(target.keys())) == 0: return fdict

    next_target = dict()
    for k, v in target.items(): 
        if isinstance(v, dict):
            for k_, v_ in v.items():
                next_target[k+"-"+k_] = v_ 
        else: 
            fdict[k] = v 
    return _to_flat_dict(next_target, fdict)


def _get_metrics_path(mode: str):
    os.makedirs("metrics", exist_ok=True)
    return f"metrics/{mode}.csv"
