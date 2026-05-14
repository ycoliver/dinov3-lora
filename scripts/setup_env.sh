#!/usr/bin/env bash
# =====================================================================
#  setup_env.sh — create / activate a local .venv and install deps
#
#  Run this ONCE before train_oneclick.sh:
#      bash scripts/setup_env.sh
#
#  Subsequent shells just need:
#      source .venv/bin/activate
# =====================================================================
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

VENV_DIR=".venv"
REQ_FILE="requirements.txt"

# Use Tsinghua mirror by default; override by exporting PIP_INDEX_URL=...
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

echo "==== DINOv3 matching env setup ============================"
echo "  Project root : $PROJECT_ROOT"
echo "  Venv dir     : $VENV_DIR"
echo "  Requirements : $REQ_FILE"
echo "  PIP index    : $PIP_INDEX_URL"
echo "============================================================"

# 1) create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creating venv..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/3] venv already exists, reusing."
fi

# 2) upgrade pip + install deps inside the venv (via mirror)
echo "[2/3] Upgrading pip (using $PIP_INDEX_URL)..."
"$VENV_DIR/bin/python" -m pip install --upgrade \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    pip wheel setuptools

echo "[3/3] Installing requirements (using $PIP_INDEX_URL)..."
"$VENV_DIR/bin/python" -m pip install \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    -r "$REQ_FILE"

echo ""
echo "==== Versions =============================================="
"$VENV_DIR/bin/python" - <<'PY'
import torch, transformers, safetensors, numpy, cv2
print(f"  torch         : {torch.__version__}")
print(f"  transformers  : {transformers.__version__}")
print(f"  safetensors   : {safetensors.__version__}")
print(f"  numpy         : {numpy.__version__}")
print(f"  opencv        : {cv2.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.backends, 'mps'):
    print(f"  MPS  available: {torch.backends.mps.is_available()}")
PY
echo "============================================================"
echo ""
echo "Activate the env in the current shell with:"
echo "    source $VENV_DIR/bin/activate"
