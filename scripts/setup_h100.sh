#!/usr/bin/env bash
set -euo pipefail

# Minimal setup for H100 box (Ubuntu). Requires CUDA-capable drivers preinstalled.

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
PYTHON=${PYTHON:-python3}
VENV_DIR="${REPO_ROOT}/.venv"

echo "[1/6] Installing system packages (ffmpeg, git)..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y ffmpeg git
else
  echo "apt-get not found; please ensure ffmpeg and git are installed. Skipping."
fi

echo "[2/6] Creating virtual environment..."
${PYTHON} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip wheel

echo "[3/6] Installing Python dependencies..."
pip install -e "${REPO_ROOT}"

echo "[4/6] Optional video IO extras (decord/av/opencv)." 
pip install decord av soundfile opencv-python || true

echo "[5/6] Building chat datasets..."
export PYTHONPATH="${REPO_ROOT}"
python -m src.data_utils.build_sft_dataset

echo "[6/6] Writing default DeepSpeed config..."
DS_CFG="${REPO_ROOT}/deepspeed_config.json"
cat > "${DS_CFG}" <<'JSON'
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "overlap_comm": true,
    "reduce_scatter": true
  },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
JSON

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
echo "Train:  PYTHONPATH=${REPO_ROOT} torchrun --standalone --nproc_per_node=1 -m src.train.train_sft --deepspeed ${DS_CFG}"
echo "Eval:   PYTHONPATH=${REPO_ROOT} python -m src.eval.infer --model_path ${REPO_ROOT}/models/omnivinci"


