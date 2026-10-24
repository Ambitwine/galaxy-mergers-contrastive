"""NNCLR arm (Dwibedi et al. 2021): same encoder, augmentation pipeline and
data loading as the SimCLR arm (see common.py). Each view is contrasted
against the nearest neighbour of the other view in a queue of past
embeddings. Hyperparameters and the train/val split are set in
config_nnclr.py and differ from the SimCLR arm.

    python -u nnclr_arm.py
"""
from __future__ import annotations
import json
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import config_nnclr as C
from common import (DEVICE, GalaxyAugment, ProjectionHead, ResNetEncoder,
                    atomic_save, encode, make_loader, stage_array, to_model_dim)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class NNCLR(nn.Module):
    def __init__(self):
        super().__init__()
        # this arm was trained with PyTorch's default initialisation
        self.encoder = ResNetEncoder(kaiming_init=False)
        self.proj = ProjectionHead(self.encoder.rep_dim, C.PROJ_HIDDEN, C.PROJ_DIM)
        self.register_buffer("queue",
                             F.normalize(torch.randn(C.QUEUE_SIZE, C.PROJ_DIM), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def embed(self, x):
        return F.normalize(self.proj(self.encoder(x)), dim=1)

    @torch.no_grad()
    def _enqueue(self, z):
        b = z.size(0)
        ptr = int(self.queue_ptr)
        if ptr + b <= C.QUEUE_SIZE:
            self.queue[ptr:ptr + b] = z
        else:
            first = C.QUEUE_SIZE - ptr
            self.queue[ptr:] = z[:first]
            self.queue[:b - first] = z[first:]
        self.queue_ptr[0] = (ptr + b) % C.QUEUE_SIZE

    @torch.no_grad()
    def _nn(self, z):
        """Cosine nearest neighbour of each row of z in the queue."""
        idx = (z.float() @ self.queue.t()).argmax(dim=1)
        return self.queue[idx]

    def loss(self, z1, z2, update_queue=True):
        """InfoNCE with NN(z1) as the anchor and z2 as positive; the other z2
        in the batch are negatives. Symmetrised if C.SYMMETRIC. fp32."""
        tau = C.TEMPERATURE
        z1f, z2f = z1.float(), z2.float()
        labels = torch.arange(z1f.size(0), device=z1f.device)
        l = F.cross_entropy((self._nn(z1f) @ z2f.t()) / tau, labels)
        if C.SYMMETRIC:
            l = 0.5 * (l + F.cross_entropy((self._nn(z2f) @ z1f.t()) / tau, labels))
        if update_queue:
            self._enqueue(z1f.detach())
        return l


def train():
    torch.manual_seed(C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    use_amp = C.USE_AMP and DEVICE.type == "cuda"
    array, mode = stage_array(C.IMAGES_NPY, C.RAM_BUDGET_GB)
    n = len(array)
    perm = np.random.default_rng(C.RANDOM_SEED).permutation(n)
    n_val = max(C.BATCH_SIZE, int(n * C.VAL_FRAC))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    log.info(f"device={DEVICE} | mode={mode} | train={len(train_idx):,} val={len(val_idx):,}")

    tr_loader = make_loader(array, train_idx, mode, C.BATCH_SIZE, shuffle=True,
                            seed=C.RANDOM_SEED, num_workers=C.NUM_WORKERS)
    va_loader = make_loader(array, val_idx, mode, C.BATCH_SIZE, shuffle=False,
                            seed=C.RANDOM_SEED + 1, num_workers=C.NUM_WORKERS)

    model = NNCLR().to(DEVICE)
    log.info(f"NNCLR params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    aug = GalaxyAugment(rotation=C.AUG_ROTATION, flip=C.AUG_FLIP,
                        translate_frac=C.AUG_TRANSLATE_FRAC, contrast=C.AUG_CONTRAST,
                        brightness=C.AUG_BRIGHTNESS, noise_sigma=C.AUG_NOISE_SIGMA)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE,
                            weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.N_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, no_imp, hist, start_epoch = float("inf"), 0, [], 1
    if C.CKPT_PATH.exists():
        ck = torch.load(C.CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, no_imp, hist = ck["best"], ck["no_imp"], ck["hist"]
        start_epoch = ck["epoch"] + 1
        log.info(f"resuming at epoch {start_epoch}")

    for ep in range(start_epoch, C.N_EPOCHS + 1):
        # linear warm-up per epoch, then cosine decay
        if ep <= C.WARMUP_EPOCHS:
            for g in opt.param_groups:
                g["lr"] = C.LEARNING_RATE * ep / C.WARMUP_EPOCHS

        model.train()
        t0 = time.time()
        tot = torch.zeros((), device=DEVICE); nb = 0
        for batch in tr_loader:
            x = to_model_dim(batch.to(DEVICE, non_blocking=True), C.IMAGE_DIM)
            v1, v2 = aug(x), aug(x)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model.loss(model.embed(v1), model.embed(v2))
            scaler.scale(loss).backward()
            if C.GRAD_CLIP > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            tot += loss.detach(); nb += 1
        if ep > C.WARMUP_EPOCHS:
            sched.step()
        train_loss = (tot / max(nb, 1)).item()

        model.eval()
        vtot = torch.zeros((), device=DEVICE); vnb = 0
        with torch.no_grad():
            for batch in va_loader:
                x = to_model_dim(batch.to(DEVICE, non_blocking=True), C.IMAGE_DIM)
                if x.shape[0] < 2:
                    continue
                with torch.amp.autocast("cuda", enabled=use_amp):
                    vtot += model.loss(model.embed(aug(x)), model.embed(aug(x)),
                                       update_queue=False).detach()
                vnb += 1
        val_loss = (vtot / max(vnb, 1)).item()

        log.info(f"epoch {ep:>3}/{C.N_EPOCHS} | train {train_loss:.4f} | "
                 f"val {val_loss:.4f} | {time.time() - t0:.1f}s")
        hist.append({"epoch": ep, "train": train_loss, "val": val_loss})
        with open(C.HISTORY_JSON, "w") as f:
            json.dump(hist, f)

        if val_loss < best - 1e-4:
            best, no_imp = val_loss, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val": val_loss},
                       C.WEIGHTS_BEST)
        else:
            no_imp += 1

        # model.state_dict() includes the queue, so resuming restores it too
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": ep, "best": best, "no_imp": no_imp, "hist": hist},
                    C.CKPT_PATH)

        if no_imp >= C.EARLY_STOP_PATIENCE and ep > C.WARMUP_EPOCHS:
            log.info(f"early stop at epoch {ep}")
            break
    log.info(f"best val {best:.4f} -> {C.WEIGHTS_BEST}")


def extract_features():
    """Encode every image with the best encoder, then fit PCA on the training
    rows and keep enough components for PCA_VAR_TARGET (2 to N_COMPONENTS_MAX)."""
    ckpt = torch.load(C.WEIGHTS_BEST, map_location=DEVICE, weights_only=False)
    model = NNCLR().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    array, _ = stage_array(C.IMAGES_NPY, C.RAM_BUDGET_GB)
    n = len(array)
    feats = encode(model.encoder, array, C.BATCH_SIZE, C.IMAGE_DIM)
    np.save(C.LATENTS_NPY, feats)
    log.info(f"latents -> {C.LATENTS_NPY}: {feats.shape}")

    labels = np.load(C.LABELS_NPY)
    train_mask = (labels["subfolder"] == "train") if "subfolder" in (labels.dtype.names or ()) \
        else np.ones(n, dtype=bool)
    k = min(C.N_COMPONENTS_MAX, feats.shape[1])
    pca = PCA(n_components=k, random_state=C.RANDOM_SEED).fit(feats[train_mask])
    cum = np.cumsum(pca.explained_variance_ratio_)
    keep = min(max(int(np.searchsorted(cum, C.PCA_VAR_TARGET) + 1), 2), k)
    fp = pca.transform(feats)[:, :keep].astype(np.float32)
    np.save(C.FEATURES_NPY, fp)
    log.info(f"PCA: kept {keep}/{k} components -> {C.FEATURES_NPY}: {fp.shape}")


if __name__ == "__main__":
    train()
    extract_features()
