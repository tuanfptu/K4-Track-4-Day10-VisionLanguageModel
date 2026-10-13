#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Không thấy nvidia-smi; chọn container có NVIDIA GPU và bật RTX 3090." >&2
  exit 1
fi
nvidia-smi
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch không thấy CUDA. Kiểm tra image và GPU passthrough.")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m pip install -r requirements-colab.txt
python scripts/prepare_data.py --zip archive.zip
python -m student.run_lab all "$@"
