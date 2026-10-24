"""NNCLR arm config: paths, model, augmentation and training settings."""
from __future__ import annotations
import os
from pathlib import Path

DATA_ROOT     = Path(os.environ.get("NNCLR_DATA_ROOT", "/scratch/s4743091/tng"))
PROCESSED_DIR = DATA_ROOT / "processed_tng"
IMAGES_NPY    = PROCESSED_DIR / "images_tng.npy"
LABELS_NPY    = PROCESSED_DIR / "labels_tng.npy"

MODELS_DIR    = DATA_ROOT / "models" / "nnclr_tng"
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

LATENTS_NPY   = MODELS_DIR / "latents_nnclr.npy"
FEATURES_NPY  = MODELS_DIR / "features_pca_nnclr.npy"
WEIGHTS_BEST  = MODELS_DIR / "nnclr_best.pt"
CKPT_PATH     = MODELS_DIR / "nnclr_last.ckpt"
HISTORY_JSON  = MODELS_DIR / "training_history_nnclr.json"

# ---- data / model ----------------------------------------------------------
IMAGE_DIM    = 128
PROJ_HIDDEN  = 512
PROJ_DIM     = 256

# ---- NNCLR -----------------------------------------------------------------
TEMPERATURE  = 0.1
QUEUE_SIZE   = 32768    # support set, ~7% of the training stack
SYMMETRIC    = True

# ---- training --------------------------------------------------------------
BATCH_SIZE          = 512
N_EPOCHS            = 300
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-6
WARMUP_EPOCHS       = 10
EARLY_STOP_PATIENCE = 30
VAL_FRAC            = 0.05
GRAD_CLIP           = 5.0
USE_AMP             = True
RANDOM_SEED         = 0
NUM_WORKERS         = 8

RAM_BUDGET_GB = 40

# ---- PCA ----
PCA_VAR_TARGET   = 0.999
N_COMPONENTS_MAX = 25

# ---- augmentations ---------------------------------------------------------
AUG_ROTATION       = True
AUG_FLIP           = True
AUG_TRANSLATE_FRAC = 0.06
AUG_CONTRAST       = (0.85, 1.15)
AUG_BRIGHTNESS     = 0.05
AUG_NOISE_SIGMA    = 0.02
