#!/bin/bash
# WorldSplat: Full Inference Pipeline
# Runs all three stages: 4D-Aware Diffusion -> GS Decoder -> Enhanced Diffusion

set -e

export PYTHONPATH=$(dirname $(dirname $(realpath $0)))
cd $PYTHONPATH

# Model paths (modify these)
CKPT_STAGE1="pretrained/worldsplat_stage1.pt"
CKPT_STAGE2="pretrained/worldsplat_stage2.pt"
CKPT_GS_DECODER="pretrained/worldsplat_gs_decoder.pt"
VAE_PRETRAINED="pretrained/OpenSora-VAE-v1.2"

# Data settings
DATA_JSON="data/nuscenes/json/nuscenes_workshop_val_seq.json"
OUTPUT_DIR="outputs/inference"
SHIFT_METERS=2.0

echo "Starting WorldSplat inference pipeline"
echo "Shift: +/- ${SHIFT_METERS}m"
echo "Output: ${OUTPUT_DIR}"

python tools/inference.py \
    --config_stage1 configs/diffusion_4d_aware.yaml \
    --config_stage2 configs/diffusion_enhanced.yaml \
    --ckpt_stage1 ${CKPT_STAGE1} \
    --ckpt_stage2 ${CKPT_STAGE2} \
    --ckpt_gs_decoder ${CKPT_GS_DECODER} \
    --vae_pretrained ${VAE_PRETRAINED} \
    --data_json ${DATA_JSON} \
    --output_dir ${OUTPUT_DIR} \
    --shift_meters ${SHIFT_METERS} \
    --num_sampling_steps 8

echo "Inference completed! Results saved to ${OUTPUT_DIR}"
