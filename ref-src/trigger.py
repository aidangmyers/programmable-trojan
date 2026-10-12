# THIS IS BASICALLY ALL FROM AiSPY!!!!!
# -*- coding: utf-8 -*-
"""
Dataset-agnostic backdoor triggers (circle, square, rectangle) + poisoning wrapper.

- Triggers operate on PIL.Image (RGB), with pixel or fractional sizing
"""

#from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Callable, Dict, Any, Union
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

# ------------------------------
# RNG helpers (reproducibility)
# ------------------------------
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def worker_init_fn_builder(base_seed: int):
    def _init(worker_id: int):
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return _init

# ------------------------------
# Trigger base
# ------------------------------
class Trigger:
    """Base trigger interface operating on PIL.Image (RGB)."""
    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        raise NotImplementedError

def _resolve_xy(position: Union[str, Tuple[int,int]], w: int, h: int, pw: int, ph: int, margin: int) -> Tuple[int,int]:
    """Top-left (x,y) for a patch of size (pw,ph)."""
    if isinstance(position, str):
        x = w - pw - margin if "right" in position else margin
        y = h - ph - margin if "bottom" in position else margin
    else:
        x, y = position
    return x, y

def _apply_jitter(x: int, y: int, jitter: int, rng: np.random.Generator) -> Tuple[int,int]:
    if jitter > 0:
        x += int(rng.integers(-jitter, jitter + 1))
        y += int(rng.integers(-jitter, jitter + 1))
    return x, y

# ------------------------------
# Rectangle trigger
# ------------------------------
@dataclass
class RectanglePatchTrigger(Trigger):
    width_px: Optional[int] = None        # one of (width_px,height_px) or (width_frac,height_frac)
    height_px: Optional[int] = None
    width_frac: Optional[float] = None    # fraction of image width/height
    height_frac: Optional[float] = None
    position: Union[str, Tuple[int,int]] = "bottom_right"
    color: Tuple[int,int,int] = (255, 255, 0)
    margin: int = 2
    jitter: int = 0

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size

        # Resolve size
        if self.width_px is not None and self.height_px is not None:
            pw, ph = self.width_px, self.height_px
        elif self.width_frac is not None and self.height_frac is not None:
            pw = max(1, int(w * float(self.width_frac)))
            ph = max(1, int(h * float(self.height_frac)))
        else:
            raise ValueError("Specify either (width_px,height_px) or (width_frac,height_frac).")

        x, y = _resolve_xy(self.position, w, h, pw, ph, self.margin)
        x, y = _apply_jitter(x, y, self.jitter, rng)

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x1 + pw), min(h, y1 + ph)

        arr = np.array(img, copy=True)
        arr[y1:y2, x1:x2] = np.array(self.color, dtype=arr.dtype)
        return Image.fromarray(arr)

# ------------------------------
# Square trigger (convenience)
# ------------------------------
@dataclass
class SquarePatchTrigger(Trigger):
    size_px: Optional[int] = None
    size_frac: Optional[float] = None     # fraction of min(w,h)
    position: Union[str, Tuple[int,int]] = "bottom_right"
    color: Tuple[int,int,int] = (255, 255, 255)
    margin: int = 2
    jitter: int = 0

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        if self.size_px is not None:
            s = self.size_px
        elif self.size_frac is not None:
            s = max(1, int(min(w, h) * float(self.size_frac)))
        else:
            raise ValueError("Specify either size_px or size_frac.")

        rect = RectanglePatchTrigger(
            width_px=s, height_px=s,
            position=self.position, color=self.color,
            margin=self.margin, jitter=self.jitter
        )
        return rect(img, rng)
