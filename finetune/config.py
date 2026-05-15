"""
Hyperparameters and configuration for DINOv2 fine-tuning.
"""

from pathlib import Path
import argparse

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "dinov3_weights"
DATASETS_DIR = PROJECT_ROOT / "datasets"
DEFAULT_CHECKPOINT = WEIGHTS_DIR / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

# ── Model ────────────────────────────────────────────────────────────
VIT_ARCH = "vit_large"         # ViT-L/16
VIT_PATCH_SIZE = 16
EMBED_DIM = 1024               # ViT-L hidden dim
PROJ_DIM = 256                 # projection head output dim
NUM_BLOCKS = 24                # total transformer blocks in ViT-L
FREEZE_BLOCKS = 22             # freeze blocks 0-21, train 22-23 (RTX 5060 8GB)

# ── Training (tuned for RTX 5060 8GB VRAM) ───────────────────────────
BATCH_SIZE = 1                 # pairs per batch (8GB VRAM limit)
NUM_EPOCHS = 15
LR_BACKBONE = 1e-5             # learning rate for unfrozen backbone blocks
LR_PROJ_HEAD = 1e-4            # learning rate for projection head
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
TEMPERATURE = 0.07             # InfoNCE temperature
IMG_SIZE = 448                 # 448 = 28*16, safe for 8GB VRAM
NUM_WORKERS = 0                # Windows compatibility
SAVE_EVERY = 1                 # save checkpoint every N epochs (每个 epoch 一次)
LOG_EVERY = 10                 # print loss every N steps

# ── Correspondence ───────────────────────────────────────────────────
EPIPOLAR_THRESH = 5e-4         # threshold for valid epipolar match
REPROJ_THRESH = 4.0            # reprojection error threshold in pixels
MAX_CORRESPONDENCES = 256      # max correspondences per pair for training
MIN_CORRESPONDENCES = 16       # skip pair if fewer correspondences

# ── MNN Matching ─────────────────────────────────────────────────────
MNN_SCORE_THRESH = 0.0         # minimum score for MNN match
FEATURE_LAYER = -1             # which ViT block's output to use (-1 = last)


def get_train_args():
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DINOv2 for local feature matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="Path to pre-trained DINOv2 weights")
    parser.add_argument("--train_pairs", type=str, required=True,
                        help="Path to training pairs file (same format as *_with_gt.txt)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the image dataset")
    parser.add_argument("--depth_root", type=str, default="",
                        help="Root directory for depth maps (optional, for ScanNet)")
    parser.add_argument("--output_dir", type=str, default="finetune_output",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--lr_proj", type=float, default=LR_PROJ_HEAD)
    parser.add_argument("--freeze_blocks", type=int, default=FREEZE_BLOCKS,
                        help="Number of blocks to freeze (0 = train all)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    return parser.parse_args()


def get_extract_args():
    """Parse feature extraction / matching arguments."""
    parser = argparse.ArgumentParser(
        description="Extract features and perform MNN matching with fine-tuned DINOv2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--pairs", type=str, required=True,
                        help="Path to pairs file (*_with_gt.txt)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the image dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save CSV matching results")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--eval_resize", type=int, nargs=2, default=[640, 480],
                        help="Evaluation resize dimensions (W H). Must match evaluate script's --resize.")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()
