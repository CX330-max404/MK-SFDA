#!/bin/bash

# ==============================================================================
# 0. Environment Setup
# ==============================================================================
# Assumes conda environment is active
# source activate YOUR_ENV_NAME (if needed)

DATA_ROOT="/ai/data/inputs"
OUTPUT_ROOT="outputs"
PRETRAIN_DATASET="isic18"
PRETRAIN_EXP_NAME="${PRETRAIN_DATASET}_MK_UNet_Source"

# ==============================================================================
# 1. Source Domain Pretraining (Step 1)
# ==============================================================================
echo "----------------------------------------------------------------"
echo "Starting Source Pretraining on ${PRETRAIN_DATASET}..."
echo "----------------------------------------------------------------"

# python train_mk.py \
#     --dataset ${PRETRAIN_DATASET} \
#     --name ${PRETRAIN_EXP_NAME} \
#     --arch MK_UNet \
#     --data_dir ${DATA_ROOT} \
#     --epochs 100 \
#     --batch_size 8 \
#     --lr 0.0001 \
#     --input_list 128,160,256 \
#     --output_dir ${OUTPUT_ROOT}

# Path to the best source model logic (modify if train_mk.py saves differently)
SOURCE_CKPT="${OUTPUT_ROOT}/${PRETRAIN_EXP_NAME}/model.pth"

if [ ! -f "$SOURCE_CKPT" ]; then
    echo "Error: Source checkpoint not found at $SOURCE_CKPT"
    # Fallback for demonstration if training didn't run or file naming is different
    # SOURCE_CKPT="outputs/${PRETRAIN_EXP_NAME}/latest_checkpoint.pth" 
fi

echo "Source Model Layout: ${SOURCE_CKPT}"

# ==============================================================================
# 2. Adaptation Step (SFDA) for Target Datasets
# ==============================================================================

# Ensure we use GPU
export CUDA_VISIBLE_DEVICES=0

# List of target datasets to adapt to
TARGET_DATASETS=("busi")
# Note: isic18 is source, so we skip it or do self-adaptation if desired.

for TARGET in "${TARGET_DATASETS[@]}"
do
    echo "----------------------------------------------------------------"
    echo "Starting Adaptation for Target: ${TARGET}"
    echo "----------------------------------------------------------------"

    # Hyperparameters for SFDA (Tuned as per Step 6 Guidelines)
    # High label_threshold (0.95+) helps HD95 by refining edges
    
    python train_sfda.py \
        --dataset ${TARGET} \
        --name "SFDA_Adapt_to_${TARGET}" \
        --pretrained_ckpt "/ai/data/Seg_MK_Unet-main/Seg_MK_Unet-main/outputs/isic18_MK_UNet_Source/model.pth" \
        --data_dir "/ai/data/inputs" \
        --output_dir ${OUTPUT_ROOT} \
        --arch MK_UNet \
        --epochs 100 \
        --batch_size 8 \
        --lr_filter 1e-4 \
        --lr_student 1e-4 \
        --input_list 128,160,256 \
        --alpha_0 1.0 \
        --alpha_1 0.01 \
        --alpha_2 0.1 \
        --alpha_3 1.0 \
        --label_threshold 0.60
        
done

echo "All Adaptations Completed."
