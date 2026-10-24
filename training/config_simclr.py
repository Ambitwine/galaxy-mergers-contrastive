"""SimCLR arm config: paths, model, augmentation and training settings."""
from __future__ import annotations
import os
from pathlib import Path

DATA_ROOT     = Path(os.environ.get("SIMCLR_DATA_ROOT", "/scratch/s4743091/tng"))
PROCESSED_DIR = DATA_ROOT / "processed_tng"
IMAGES_NPY    = PROCESSED_DIR / "images_tng.npy"   # (N, 128, 128) float32 in [0,1]
LABELS_NPY    = PROCESSED_DIR / "labels_tng.npy"   # structured: is_major_merger, subfolder, ...

MODELS_DIR = DATA_ROOT / "models" / "simclr_tng"
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

# ---- Image geometry ---------------------------------------------------------
IMAGE_DIM   = 128
IN_CHANNELS = 1

# ---- Encoder / projection ---------------------------------------------------
RESNET_LAYERS = (2, 2, 2, 2)     # ResNet-18
RESNET_WIDTH  = 64               # 512-d features
PROJ_HIDDEN   = 512
PROJ_DIM      = 128

# ---- Contrastive loss -------------------------------------------------------
NT_XENT_TEMP = 0.2

# ---- Augmentations ----------------------------------------------------------
AUG_ROTATION       = True
AUG_FLIP           = True
AUG_TRANSLATE_FRAC = 0.06
AUG_NOISE_SIGMA    = 0.02
AUG_CONTRAST       = (0.85, 1.15)
AUG_BRIGHTNESS     = 0.03

# ---- Training ---------------------------------------------------------------
BATCH_SIZE          = 512
N_EPOCHS            = 300
LEARNING_RATE       = 3e-3
WEIGHT_DECAY        = 1e-4
WARMUP_EPOCHS       = 10
EARLY_STOP_PATIENCE = 30
RANDOM_SEED         = 42
USE_AMP             = True
GRAD_CLIP           = 5.0
NUM_WORKERS         = 8
VAL_FRAC_FALLBACK   = 0.1

# stacks larger than this are memmapped from $TMPDIR instead (keep below --mem)
RAM_BUDGET_GB = 40

# ---- PCA --------------------------------------------------------------------
PCA_VARIANCE_TARGET = 0.999
N_COMPONENTS_MAX    = 25

# ---- Output filenames -------------------------------------------------------
BEST_ENCODER  = MODELS_DIR / "simclr_encoder_best.pt"
CKPT_PATH     = MODELS_DIR / "simclr_tng.ckpt"
HISTORY_JSON  = MODELS_DIR / "training_history.json"
LATENTS_NPY   = MODELS_DIR / "latents_simclr.npy"
PCA_JOBLIB    = MODELS_DIR / "pca_simclr.joblib"
FEATURES_NPY  = MODELS_DIR / "features_pca_simclr.npy"
