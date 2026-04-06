#!/bin/bash
# Download publicly referenced pretrained models into pretrained/

set -euo pipefail

REPO_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
cd "$REPO_ROOT"

OUT_DIR=${OUT_DIR:-pretrained}
HF_TOKEN=${HF_TOKEN:-}

mkdir -p "$OUT_DIR"

echo "Downloading pretrained assets into: $OUT_DIR"

uv run --with huggingface_hub python - "$OUT_DIR" "$HF_TOKEN" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

out_dir = sys.argv[1]
hf_token = sys.argv[2] or None

models = [
    ("hpcai-tech/OpenSora-VAE-v1.2", "OpenSora-VAE-v1.2"),
    ("google/t5-v1_1-xxl", "t5-v1_1-xxl"),
    ("stabilityai/sd-vae-ft-ema", "sd-vae-ft-ema"),
]

for repo_id, subdir in models:
    local_dir = os.path.join(out_dir, subdir)
    print(f"[download] {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=hf_token,
        resume_download=True,
    )

print("")
print("Done.")
print("WorldSplat stage checkpoints are not downloaded here.")
print("The repository README still lists them as 'Coming soon'.")
PY

