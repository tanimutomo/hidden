import csv
from dataclasses import dataclass
import os
import shutil
import sys
import yaml
import typing

import comet_ml
import moment
import torch
import torchvision

sys.path.append(os.path.abspath("."))


Metrics = typing.Dict[str, float]
Images = typing.Dict[str, torch.FloatTensor]

@dataclass
class CometConfig:
    project: str
    workspace: str
    api_key: str
    resume_experiment_key: str =""

@dataclass
class Config:
    name: str
    tags: typing.Dict[str, str]
    comet: CometConfig
    use_comet: bool =False
    resume_training: bool =False

CHECKPOINT_PATH = "checkpoint.pth"
PARAMETERS_PATH = "parameters.pth"

_comet: comet_ml.Experiment = None
_cfg: Config = None
_epoch: int = 0
_epochs: int = 0
_mode: str = ""


def init(cfg: Config, epochs: int):
    global _cfg, _comet, _epochs
    _cfg = cfg
    _epochs = epochs

    if _cfg.resume_training:
        if not os.path.exists("train.log"):
            raise ValueError(f"The specified experiment is seems to be a new experiment.")
        if _cfg.use_comet and not _cfg.comet.resume_experiment_key:
            raise ValueError(f"cfg.comet.resume_experiment_key is empty.")

    if _cfg.use_comet:
        exp_args = dict(
            project_name=_cfg.comet.project,
            workspace=_cfg.comet.workspace,
            api_key=_cfg.comet.api_key,
            auto_param_logging=False,
            auto_metric_logging=False,
            parse_args=False
        )
        if _cfg.comet.resume_experiment_key:
            exp_args["previous_experiment"] = _cfg.comet.resume_experiment_key
            _comet = comet_ml.ExistingExperiment(**exp_args)
        else:
            _comet = comet_ml.Experiment(**exp_args)
            _comet.set_name(_cfg.name)

        if _cfg.tags:
            _comet.add_tags(_cfg.tags)

    print(f"Start experiment: {_cfg.name}")


def set_epoch(epoch: int):
    global _epoch
    _epoch = epoch


def set_mode(mode: str):
    global _mode
    _mode = mode


def log_hyper_parameters(params: dict):
    if _cfg.use_comet:
        _comet.log_parameters(_to_flat_dict(params, dict()))


def log_metrics(metrics: Metrics, test=False):
    stdout = f"{_mode.upper()} [{_epoch:d}/{_epochs:d}]  "

    for name, value in metrics.items():
        stdout += f"{name} = {value:.4f} / "
        if not _comet: continue
        _comet.log_metric( f"{_mode}-{name}", value, step=_epoch)

    _save_metrics_to_csv(metrics)
    print(stdout)


def log_images(imgs: Images, transformer=None):
    dirname = f"images/{_epoch}"
    os.makedirs(dirname, exist_ok=True)
    for name, img in imgs.items():
        filename = f"{_mode}_{name}.png"
        path = os.path.join(dirname, filename)
        if transformer:
            img = transformer(img)
        torchvision.utils.save_image(img, path)
        if _comet:
            _comet.log_image(path, step=_epoch)


def log_texts(texts: typing.Dict[str, str]):
    for key, text in texts.items():
        text = f"{_mode}_{key} = ({text})"
        if _comet:
            _comet.log_text(text, step=_epoch)


def log_checkpoint(ckpt: dict):
    ckpt["epoch"] = _epoch
    torch.save(ckpt, CHECKPOINT_PATH)
    if _comet:
        _send_file_to_comet(CHECKPOINT_PATH, overwrite=True)


def log_parameters(params: dict):
    torch.save(params, PARAMETERS_PATH)
    if _comet:
        _send_file_to_comet(PARAMETERS_PATH)


def get_checkpoint()-> typing.Tuple[int, dict]:
    if not _cfg.resume_training:
        raise ValueError("This training is new experiment.")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    shutil.copy(CHECKPOINT_PATH, CHECKPOINT_PATH+f".bkup.{moment.now().format('YYYY-MMDD-HHmm-ss')}")
    epoch = ckpt.pop("epoch")
    return epoch, ckpt


def get_parameters(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def _save_metrics_to_csv(_metrics: Metrics):
    metrics = dict()
    for name, value in _metrics.items():
        metrics[f"{_mode}-{name}"] = value
    metrics.update({"epoch": _epoch, "timestamp": moment.now().format("YYYY-MMDD-HHmm-ss")})

    path = _get_metrics_path()
    open_mode = "a" if os.path.exists(path) else "w"
    with open(path, open_mode) as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if open_mode == "w":
            writer.writeheader()
        writer.writerow(metrics)

    if _comet:
        _send_file_to_comet(path, overwrite=True)


def _send_file_to_comet(path :str, overwrite=False):
    _comet.log_asset(path, overwrite=overwrite, step=_epoch)


def _to_flat_dict(target :dict, fdict: dict):
    if len(list(target.keys())) == 0: return fdict

    next_target = dict()
    for k, v in target.items(): 
        if isinstance(v, dict):
            for k_, v_ in v.items():
                next_target[k+"."+k_] = v_ 
        else: 
            fdict[k] = v 
    return _to_flat_dict(next_target, fdict)


def _get_metrics_path():
    os.makedirs("metrics", exist_ok=True)
    return f"metrics/{_mode}.csv"
