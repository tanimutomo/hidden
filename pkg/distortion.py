from dataclasses import dataclass, field
import typing

import distortion


@dataclass
class Config:
    name: str
    # Dropout, Cropout, Crop, Resize
    p: float
    # GaussianBlur, Combined
    w: int
    # GaussianBlur
    s: float
    # JPEGCompression
    qf: int
    # Combined
    ps: typing.List[float]
    ss: typing.List[float]
    


def init(mean: typing.List[float], std: typing.List[float]):
    distortion.init(mean, std)


def get(cfg: Config) -> distortion.Distortioner:
    if   cfg.name == "identity": return distortion.Identity()
    elif cfg.name == "dropout": return distortion.Dropout(cfg.p)
    elif cfg.name == "cropout": return distortion.Cropout(cfg.p)
    elif cfg.name == "crop": return distortion.Crop(cfg.p)
    elif cfg.name == "resize": return distortion.Resize(cfg.p)
    elif cfg.name == "gaussian_blur": return distortion.GaussianBlur(cfg.w, cfg.s)
    elif cfg.name == "jpeg_compression": return distortion.JPEGCompression(cfg.qf)
    elif cfg.name == "jpeg_mask": return distortion.JPEGMask()
    elif cfg.name == "jpeg_drop": return distortion.JPEGDrop()
    elif cfg.name == "jpeg": return distortion.JPEGCompression(cfg.qf)
    elif cfg.name == "combined": return distortion.Combined(cfg.ps, cfg.w, cfg.ss)
    else: raise NotImplementedError
