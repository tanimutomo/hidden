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


def get(cfg: DistortionConfig) -> distortion.Distortioner:
    return {
        "identity": distortion.Identity(),
        "dropout": distoriton.Dropout(cfg.p),
        "cropout": distoriton.Cropout(cfg.p),
        "crop": distoriton.Crop(cfg.p),
        "resize": distortion.Resize(cfg.p),
        "gaussian_blur": distortion.GaussianBlur(cfg.w, cfg.s),
        "jpeg_compression": distortion.JPEGCompression(cfg.qf),
        "jpeg_mask": distortion.JPEGMask(),
        "jpeg_drop": distortion.JPEGDrop(),
    }[cfg.name]