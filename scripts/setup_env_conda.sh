#!/usr/bin/env bash
# =====================================================================
#  setup_env_conda.sh — create a conda env and install all deps
#
#  Run ONCE on a fresh machine (Linux/CUDA or macOS):
#      bash scripts/setup_env_conda.sh
#
#  Subsequent shells just need:
#      conda activate dinov3-match
#
#  Optional overrides (export before running):
#      ENV_NAME=dinov3-match     # conda env name
#      PY_VER=3.10               # python version
#      CUDA=cu121                # cu118 | cu121 | cu124 | cpu | auto
#      PIP_INDEX_URL=...         # mirror for non-torch packages
# =====================================================================
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

ENV_NAME="${ENV_NAME:-dinov3-match}"
PY_VER="${PY_VER:-3.10}"
REQ_FILE="requirements.txt"
CUDA="${CUDA:-auto}"

# Mirror for general pip packages (NOT for torch — torch goes to pytorch.org)
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

echo "==== DINOv3 matching env setup (conda) ====================="
echo "  Project root : $PROJECT_ROOT"
echo "  conda env    : $ENV_NAME"
echo "  Python       : $PY_VER"
echo "  CUDA mode    : $CUDA"
echo "  PIP index    : $PIP_INDEX_URL  (torch 用 pytorch.org)"
echo "============================================================"

# ---------------------------------------------------------------------
# 0) sanity: conda must be available
# ---------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERR] conda not found. Install Miniconda first:"
    echo "      https://docs.conda.io/projects/miniconda/en/latest/"
    exit 1
fi

# enable `conda activate` inside this script
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------
# 1) create env if missing
# ---------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[1/4] conda env '$ENV_NAME' already exists, reusing."
else
    echo "[1/4] Creating conda env '$ENV_NAME' (python=$PY_VER) ..."
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

conda activate "$ENV_NAME"
echo "       active python : $(python -V)  ($(which python))"

# ---------------------------------------------------------------------
# 2) detect CUDA version (if not specified)
# ---------------------------------------------------------------------
if [ "$CUDA" = "auto" ]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        CUDA="cpu"   # macOS -> use CPU/MPS wheel
    elif command -v nvidia-smi >/dev/null 2>&1; then
        # parse "CUDA Version: 12.1" from nvidia-smi header
        DRV_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -n1 | awk '{print $3}')"
        if [ -n "${DRV_CUDA:-}" ]; then
            MAJOR="${DRV_CUDA%%.*}"
            MINOR="${DRV_CUDA##*.}"
            if [ "$MAJOR" -ge 12 ] && [ "$MINOR" -ge 4 ]; then CUDA="cu124"
            elif [ "$MAJOR" -ge 12 ];                       then CUDA="cu121"
            elif [ "$MAJOR" -ge 11 ] && [ "$MINOR" -ge 8 ]; then CUDA="cu118"
            else CUDA="cu118"
            fi
        else
            CUDA="cu121"   # have GPU but unknown drv -> safe default
        fi
    else
        CUDA="cpu"
    fi
fi
echo "[2/4] resolved CUDA target = $CUDA"

# ---------------------------------------------------------------------
# 3) upgrade pip + install torch from official index
# ---------------------------------------------------------------------
echo "[3/4] Upgrading pip / wheel / setuptools (mirror) ..."
python -m pip install --upgrade \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    pip wheel setuptools

# Torch wheels: default to Aliyun mirror (much faster in China than pytorch.org).
# Override with TORCH_MIRROR=official to use download.pytorch.org instead.
TORCH_MIRROR="${TORCH_MIRROR:-aliyun}"
case "$TORCH_MIRROR" in
    aliyun)   TORCH_BASE="https://mirrors.aliyun.com/pytorch-wheels"
              TORCH_HOST="mirrors.aliyun.com" ;;
    official) TORCH_BASE="https://download.pytorch.org/whl"
              TORCH_HOST="download.pytorch.org" ;;
    *) echo "[ERR] unknown TORCH_MIRROR: $TORCH_MIRROR"; exit 1 ;;
esac
case "$CUDA" in
    cu118) TORCH_INDEX="$TORCH_BASE/cu118" ;;
    cu121) TORCH_INDEX="$TORCH_BASE/cu121" ;;
    cu124) TORCH_INDEX="$TORCH_BASE/cu124" ;;
    cpu)   TORCH_INDEX="$TORCH_BASE/cpu"   ;;
    *)     echo "[ERR] unknown CUDA target: $CUDA"; exit 1 ;;
esac
echo "       installing torch / torchvision from $TORCH_INDEX ..."
python -m pip install \
    --index-url "$TORCH_INDEX" \
    --trusted-host "$TORCH_HOST" \
    "torch>=2.0" "torchvision>=0.15"

# ---------------------------------------------------------------------
# 4) install the rest of requirements (mirror, but keep installed torch)
# ---------------------------------------------------------------------
echo "[4/4] Installing remaining requirements (mirror) ..."
python -m pip install \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    -r "$REQ_FILE"

# ---------------------------------------------------------------------
# version dump
# ---------------------------------------------------------------------
echo ""
echo "==== Versions =============================================="
python - <<'PY'
import torch, transformers, safetensors, numpy, cv2
print(f"  python        : {__import__('sys').version.split()[0]}")
print(f"  torch         : {torch.__version__}")
print(f"  torchvision   : {__import__('torchvision').__version__}")
print(f"  transformers  : {transformers.__version__}")
print(f"  safetensors   : {safetensors.__version__}")
print(f"  numpy         : {numpy.__version__}")
print(f"  opencv        : {cv2.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device  : {torch.cuda.get_device_name(0)}")
    print(f"  torch.cuda   : {torch.version.cuda}")
if hasattr(torch.backends, 'mps'):
    print(f"  MPS  available: {torch.backends.mps.is_available()}")
PY
echo "============================================================"
echo ""
echo "Done. Activate the env with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "Then run training:"
echo "    bash scripts/train_oneclick.sh tiny     # quick sanity"
echo "    bash scripts/train_oneclick.sh full     # full run"
