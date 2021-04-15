from dataclasses import dataclass

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


def get(cfg: Config) -> distortion.Distortioner:
    if cfg.name == "identity": return distortion.Identity()
    elif cfg.name == "dropout": return distortion.Dropout(cfg.p)
    elif cfg.name == "cropout": return distortion.Cropout(cfg.p)
    elif cfg.name == "crop": return distortion.Crop(cfg.p)
    elif cfg.name == "resize": return distortion.Resize(cfg.p)
    elif cfg.name == "gaussian_blur": return distortion.GaussianBlur(cfg.w, cfg.s)
    elif cfg.name == "jpeg_compression": return distortion.JPEGCompression(cfg.qf)
    elif cfg.name == "jpeg_mask": return distortion.JPEGMask()
    elif cfg.name == "jpeg_drop": return distortion.JPEGDrop()
    elif cfg.name == "jpeg": return distortion.JPEGCompression(cfg.qf)
    else: raise NotImplementedError