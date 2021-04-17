from dataclasses import dataclass, field
import typing

import distortion


@dataclass
class Config:
    name: str
    # Dropout, Cropout, Crop, Resize
    p: float = 0.0
    # GaussianBlur
    w: int = 0
    # GaussianBlur
    s: float = 0.0
    # JPEGCompression
    qf: int = 0
    # JPEGDrop, JPEGMask
    mean: typing.List[float] = field(default_factory=list)
    std: typing.List[float] = field(default_factory=list)


def get(cfg: Config) -> distortion.Distortioner:
    if cfg.name == "identity": return distortion.Identity()
    elif cfg.name == "dropout": return distortion.Dropout(cfg.p)
    elif cfg.name == "cropout": return distortion.Cropout(cfg.p)
    elif cfg.name == "crop": return distortion.Crop(cfg.p)
    elif cfg.name == "resize": return distortion.Resize(cfg.p)
    elif cfg.name == "gaussian_blur": return distortion.GaussianBlur(cfg.w, cfg.s)
    elif cfg.name == "jpeg_compression": return distortion.JPEGCompression(cfg.qf)
    elif cfg.name == "jpeg_mask": return distortion.JPEGMask(cfg.mean, cfg.std)
    elif cfg.name == "jpeg_drop": return distortion.JPEGDrop(cfg.mean, cfg.std)
    elif cfg.name == "jpeg": return distortion.JPEGCompression(cfg.qf)
    else: raise NotImplementedError