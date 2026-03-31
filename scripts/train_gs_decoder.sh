#!/bin/bash
# WorldSplat: Gaussian Decoder Training
# Trains the feed-forward latent Gaussian decoder

set -e

export PYTHONPATH=$(dirname $(dirname $(realpath $0)))
cd $PYTHONPATH

# Distributed training settings
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29503}

# Experiment settings
CONFIG="configs/gs_decoder.py"
SAVE_FOLDER="outputs/gs_decoder"

echo "Starting Gaussian Decoder training"
echo "Config: ${CONFIG}"
echo "Save folder: ${SAVE_FOLDER}"

accelerate launch \
    --num_processes $((GPUS * NNODES)) \
    --num_machines $NNODES \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    tools/train_gs_decoder.py \
    --config ${CONFIG} \
    --output_dir ${SAVE_FOLDER}

echo "Gaussian Decoder training completed!"
