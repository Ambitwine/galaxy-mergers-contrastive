"""Code shared by the SimCLR and NNCLR arms: augmentations, the ResNet-18
encoder, the projection head, data staging and feature extraction.
"""
from __future__ import annotations
import logging
import math
import os
import shutil
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- augmentations ----
class GalaxyAugment:
    """Batched GPU augmentations. Each call returns one random view of x
    (B, 1, H, W) in [0, 1]: random rotation, flips, a small translation,
    contrast and brightness jitter, and Gaussian noise. The settings come from
    each arm's config."""

    def __init__(self, *, rotation, flip, translate_frac, contrast, brightness,
                 noise_sigma):
        self.rotation = rotation
        self.flip = flip
        self.tfrac = translate_frac
        self.contrast = contrast
        self.brightness = brightness
        self.noise = noise_sigma

    def _geom(self, x):
        B = x.shape[0]
        dev, dt = x.device, x.dtype
        a = (torch.rand(B, device=dev, dtype=dt) * 2 * math.pi
             if self.rotation else torch.zeros(B, device=dev, dtype=dt))
        c, s = torch.cos(a), torch.sin(a)
        sx = torch.ones(B, device=dev, dtype=dt)
        sy = torch.ones(B, device=dev, dtype=dt)
        if self.flip:
            sx = torch.where(torch.rand(B, device=dev) < 0.5, -sx, sx)
            sy = torch.where(torch.rand(B, device=dev) < 0.5, -sy, sy)
        tx = (torch.rand(B, device=dev, dtype=dt) * 2 - 1) * self.tfrac
        ty = (torch.rand(B, device=dev, dtype=dt) * 2 - 1) * self.tfrac
        mat = torch.zeros(B, 2, 3, device=dev, dtype=dt)
        mat[:, 0, 0] = sx * c;  mat[:, 0, 1] = -sx * s; mat[:, 0, 2] = tx
        mat[:, 1, 0] = sy * s;  mat[:, 1, 1] = sy * c;  mat[:, 1, 2] = ty
        grid = F.affine_grid(mat, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)

    @torch.no_grad()
    def __call__(self, x):
        B = x.shape[0]
        x = self._geom(x)
        if self.contrast is not None:
            lo, hi = self.contrast
            x = x * (torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype) * (hi - lo) + lo)
        if self.brightness:
            x = x + (torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype) * 2 - 1) * self.brightness
        if self.noise:
            x = x + torch.randn_like(x) * self.noise
        return x.clamp(0.0, 1.0)


# ---- encoder (ResNet-18) ----
class BasicBlock(nn.Module):
    # parameter names used by the original NNCLR script, so its checkpoints still load
    _LEGACY = {"c1": "conv1", "b1": "bn1", "c2": "conv2", "b2": "bn2", "sc": "short"}

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in [k for k in state_dict if k.startswith(prefix)]:
            head, _, rest = key[len(prefix):].partition(".")
            if head in self._LEGACY:
                state_dict[prefix + self._LEGACY[head] + "." + rest] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.short(x), inplace=True)


class ResNetEncoder(nn.Module):
    """ResNet-18 with a small-image stem (3x3 stride 1, no max-pool).
    forward(x) -> (B, 8 * width) globally pooled features."""

    def __init__(self, in_ch=1, width=64, layers=(2, 2, 2, 2), kaiming_init=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.in_c = width
        self.layer1 = self._make(width,     layers[0], 1)
        self.layer2 = self._make(width * 2, layers[1], 2)
        self.layer3 = self._make(width * 4, layers[2], 2)
        self.layer4 = self._make(width * 8, layers[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rep_dim = width * 8
        if kaiming_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _make(self, cout, n, stride):
        blocks = [BasicBlock(self.in_c, cout, stride)]
        self.in_c = cout
        for _ in range(1, n):
            blocks.append(BasicBlock(cout, cout, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        h = self.stem(x)
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        return self.pool(h).flatten(1)


class ProjectionHead(nn.Module):
    def __init__(self, rep_dim, hidden, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


# ---- data staging and loading ----
def stage_array(src_path, budget_gb):
    """Load the stack into RAM if it fits budget_gb, otherwise copy it to
    node-local $TMPDIR and memmap it. Returns (array, "ram" | "memmap")."""
    src_path = str(src_path)
    mm = np.load(src_path, mmap_mode="r")
    gb = mm.size * mm.dtype.itemsize / 1e9
    if gb <= budget_gb:
        log.info(f"loading {gb:.1f} GB into RAM")
        return np.array(mm), "ram"
    tmp = os.environ.get("TMPDIR", tempfile.gettempdir())
    dst = os.path.join(tmp, os.path.basename(src_path))
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src_path)):
        log.info(f"copying {gb:.1f} GB to {dst}")
        shutil.copy(src_path, dst)
    return np.load(dst, mmap_mode="r"), "memmap"


class _ImageDataset(Dataset):
    """Per-sample dataset for the memmap case."""

    def __init__(self, array, indices):
        self.array = array
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = np.array(self.array[self.indices[i]], dtype=np.float32)
        return torch.from_numpy(img).unsqueeze(0)


class _RamLoader:
    """Batches sliced straight from the in-RAM array."""

    def __init__(self, array, indices, batch, shuffle, seed, drop_last):
        self.array = array
        self.indices = np.asarray(indices)
        self.batch = batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.pin = DEVICE.type == "cuda"

    def __len__(self):
        n = len(self.indices)
        return n // self.batch if self.drop_last else math.ceil(n / self.batch)

    def __iter__(self):
        idx = self.indices.copy()
        if self.shuffle:
            self.rng.shuffle(idx)
        n = len(idx)
        last = n - (n % self.batch) if self.drop_last else n
        for i in range(0, last, self.batch):
            chunk = np.ascontiguousarray(self.array[idx[i:i + self.batch]], dtype=np.float32)
            x = torch.from_numpy(chunk).unsqueeze(1)
            yield x.pin_memory() if self.pin else x


def make_loader(array, indices, mode, batch, shuffle, seed, num_workers):
    if mode == "ram":
        return _RamLoader(array, indices, batch, shuffle, seed, drop_last=shuffle)
    return DataLoader(
        _ImageDataset(array, indices),
        batch_size=batch, shuffle=shuffle, num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0), drop_last=shuffle,
        prefetch_factor=(4 if num_workers > 0 else None))


def to_model_dim(x, dim):
    if x.shape[-1] != dim:
        x = F.interpolate(x, size=(dim, dim), mode="area")
    return x


def atomic_save(obj, path):
    """torch.save to a temporary file, then rename, so an interrupted write
    never leaves a broken checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def encode(encoder, array, batch, dim):
    """Run the encoder over the whole stack and return (N, rep_dim) float32."""
    n = len(array)
    reps = np.empty((n, encoder.rep_dim), dtype=np.float32)
    for i in range(0, n, batch):
        chunk = np.array(array[i:i + batch], dtype=np.float32)
        t = to_model_dim(torch.from_numpy(chunk).unsqueeze(1).to(DEVICE), dim)
        reps[i:i + len(chunk)] = encoder(t).cpu().numpy()
        if (i // batch) % 50 == 0:
            log.info(f"  encoded {min(i + batch, n):,}/{n:,}")
    return reps
