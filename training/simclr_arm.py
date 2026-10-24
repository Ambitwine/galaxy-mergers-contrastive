"""SimCLR arm (Chen et al. 2020): NT-Xent training of the ResNet-18 encoder on
the TNG stack, then feature extraction and PCA.

    python -u simclr_arm.py
"""
from __future__ import annotations
import json
import logging
import math
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import config_simclr as C
from common import (DEVICE, GalaxyAugment, ProjectionHead, ResNetEncoder,
                    atomic_save, encode, make_loader, stage_array, to_model_dim)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def make_encoder():
    return ResNetEncoder(in_ch=C.IN_CHANNELS, width=C.RESNET_WIDTH, layers=C.RESNET_LAYERS)


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = make_encoder()
        self.head = ProjectionHead(self.encoder.rep_dim, C.PROJ_HIDDEN, C.PROJ_DIM)

    def forward(self, x):
        return self.head(self.encoder(x))


def nt_xent(z1, z2, temp=C.NT_XENT_TEMP):
    """NT-Xent loss over two batches of views, computed in fp32. The positive
    for row i is its other view; the remaining 2B - 2 rows are negatives."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0).float(), dim=1)
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def load_split():
    labels = np.load(C.LABELS_NPY)
    if "subfolder" in (labels.dtype.names or ()):
        train_mask = labels["subfolder"] == "train"
    else:
        rng = np.random.default_rng(C.RANDOM_SEED)
        train_mask = rng.random(len(labels)) >= C.VAL_FRAC_FALLBACK
    return labels, train_mask


def train():
    torch.manual_seed(C.RANDOM_SEED)
    np.random.seed(C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    labels, train_mask = load_split()
    array, mode = stage_array(C.IMAGES_NPY, C.RAM_BUDGET_GB)
    if len(array) != len(labels):
        raise RuntimeError(f"stack/labels mismatch: {len(array)} vs {len(labels)}")
    idx_tr = np.where(train_mask)[0]
    idx_va = np.where(~train_mask)[0]
    log.info(f"device={DEVICE} | mode={mode} | train={len(idx_tr):,} val={len(idx_va):,}")

    tr_loader = make_loader(array, idx_tr, mode, C.BATCH_SIZE, shuffle=True,
                            seed=C.RANDOM_SEED, num_workers=C.NUM_WORKERS)
    va_loader = make_loader(array, idx_va, mode, C.BATCH_SIZE, shuffle=False,
                            seed=C.RANDOM_SEED + 1, num_workers=C.NUM_WORKERS)

    model = SimCLR().to(DEVICE)
    log.info(f"SimCLR params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    aug = GalaxyAugment(rotation=C.AUG_ROTATION, flip=C.AUG_FLIP,
                        translate_frac=C.AUG_TRANSLATE_FRAC, contrast=C.AUG_CONTRAST,
                        brightness=C.AUG_BRIGHTNESS, noise_sigma=C.AUG_NOISE_SIGMA)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE,
                            weight_decay=C.WEIGHT_DECAY)
    steps_per_epoch = max(1, len(tr_loader))
    total_steps = C.N_EPOCHS * steps_per_epoch
    warmup = C.WARMUP_EPOCHS * steps_per_epoch

    def lr_at(step):
        # linear warm-up, then cosine decay, per step
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    use_amp = C.USE_AMP and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = {"train_loss": [], "val_loss": []}
    best_val, bad, gstep, start_epoch = float("inf"), 0, 0, 1

    if C.CKPT_PATH.exists():
        # checkpoints from the earlier revision hold only model, opt, epoch and gstep
        ck = torch.load(C.CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        gstep = ck.get("gstep", 0)
        best_val = ck.get("best_val", float("inf"))
        bad = ck.get("bad", 0)
        if "history" in ck:
            history = ck["history"]
        elif C.HISTORY_JSON.exists():
            history = json.loads(C.HISTORY_JSON.read_text())
        start_epoch = ck["epoch"] + 1
        log.info(f"resuming at epoch {start_epoch}")

    for ep in range(start_epoch, C.N_EPOCHS + 1):
        model.train(); t0 = time.time()
        tl = torch.zeros((), device=DEVICE); tn = 0
        for batch in tr_loader:
            x = to_model_dim(batch.to(DEVICE, non_blocking=True), C.IMAGE_DIM)
            for g in opt.param_groups:
                g["lr"] = C.LEARNING_RATE * lr_at(gstep)
            v1, v2 = aug(x), aug(x)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = nt_xent(model(v1), model(v2))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(opt); scaler.update()
            tl += loss.detach() * x.shape[0]; tn += x.shape[0]; gstep += 1
        train_loss = (tl / max(tn, 1)).item()

        model.eval()
        vl = torch.zeros((), device=DEVICE); vn = 0
        with torch.no_grad():
            for batch in va_loader:
                x = to_model_dim(batch.to(DEVICE, non_blocking=True), C.IMAGE_DIM)
                if x.shape[0] < 2:
                    continue
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = nt_xent(model(aug(x)), model(aug(x)))
                vl += loss.detach() * x.shape[0]; vn += x.shape[0]
        val_loss = (vl / max(vn, 1)).item()
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        log.info(f"epoch {ep:>3}/{C.N_EPOCHS} | train {train_loss:.4f} | "
                 f"val {val_loss:.4f} | {time.time() - t0:.1f}s")

        if val_loss < best_val - 1e-4:
            best_val, bad = val_loss, 0
            torch.save({"encoder": model.encoder.state_dict(),
                        "rep_dim": model.encoder.rep_dim, "epoch": ep,
                        "val_loss": val_loss, "image_dim": C.IMAGE_DIM},
                       C.BEST_ENCODER)
        else:
            bad += 1

        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": ep, "gstep": gstep,
                     "best_val": best_val, "bad": bad, "history": history},
                    C.CKPT_PATH)
        C.HISTORY_JSON.write_text(json.dumps(history, indent=2))

        if bad >= C.EARLY_STOP_PATIENCE:
            log.info(f"early stop at epoch {ep} (best val {best_val:.4f})")
            break
    log.info(f"best encoder -> {C.BEST_ENCODER}")


def extract_features(weights=None):
    """Encode every image with the best encoder, then fit PCA on the training
    rows, keeping enough components for PCA_VARIANCE_TARGET (at most
    N_COMPONENTS_MAX)."""
    weights = weights or C.BEST_ENCODER
    ckpt = torch.load(weights, map_location=DEVICE, weights_only=False)
    enc = make_encoder().to(DEVICE).eval()
    enc.load_state_dict(ckpt["encoder"])
    log.info(f"loaded encoder from {weights} (epoch {ckpt.get('epoch')})")

    _, train_mask = load_split()
    array, _ = stage_array(C.IMAGES_NPY, C.RAM_BUDGET_GB)
    reps = encode(enc, array, C.BATCH_SIZE, C.IMAGE_DIM)
    np.save(C.LATENTS_NPY, reps)
    log.info(f"latents -> {C.LATENTS_NPY}: {reps.shape}")

    tr = reps[train_mask]
    cum = np.cumsum(PCA(n_components=min(tr.shape)).fit(tr).explained_variance_ratio_)
    n_keep = min(int(np.searchsorted(cum, C.PCA_VARIANCE_TARGET) + 1), C.N_COMPONENTS_MAX)
    pca = PCA(n_components=n_keep).fit(tr)
    joblib.dump(pca, C.PCA_JOBLIB)
    feats = pca.transform(reps).astype(np.float32)
    np.save(C.FEATURES_NPY, feats)
    log.info(f"PCA: {n_keep} components -> {C.FEATURES_NPY}: {feats.shape}")
    return feats


if __name__ == "__main__":
    train()
    extract_features()
