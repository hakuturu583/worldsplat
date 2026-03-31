#!/bin/bash
# WorldSplat Stage 2: Enhanced Diffusion Model Training
# Refines novel-view renderings from Gaussian splatting

set -e

export PYTHONPATH=$(dirname $(dirname $(realpath $0)))
cd $PYTHONPATH

# Distributed training settings
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29502}

# Experiment settings
EXP_NAME="worldsplat_stage2"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONFIG="configs/diffusion_enhanced.yaml"
SAVE_FOLDER="outputs/${EXP_NAME}_${TIMESTAMP}"

# Pretrained model paths (modify these)
VAE_PRETRAINED="pretrained/OpenSora-VAE-v1.2"
CHECKPOINT="pretrained/worldsplat_stage2.pt"

echo "Starting Stage 2 training: Enhanced Diffusion Model"
echo "Config: ${CONFIG}"
echo "Save folder: ${SAVE_FOLDER}"
echo "Nodes: ${NNODES}, GPUs per node: ${GPUS}"

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$MASTER_PORT \
    tools/train_diffusion.py \
    --stage 2 \
    --yaml_file ${CONFIG} \
    --batch_size 1 \
    --num_workers 4 \
    --save_folder ${SAVE_FOLDER} \
    --vae_pretrained ${VAE_PRETRAINED} \
    --private_ckpt_path ${CHECKPOINT} \
    --epochs 1000 \
    --lr 1e-4 \
    --warmup_steps 1500 \
    --ckpt_every 50

echo "Stage 2 training completed!"
