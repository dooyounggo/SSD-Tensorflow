#!/bin/bash

echo "Enter subdirectory name:"
read -r input

if [ -z "${DATASET_ROOT}" ]
then
  DATASET_ROOT="${HOME}/Projects/datasets"
fi

if [ -n "${input}" ]
then
    DATASET_DIR="${DATASET_ROOT}/VOC2007_${input}/JPEGImages"
else
    DATASET_DIR="${DATASET_ROOT}/VOC2007/JPEGImages"
fi
EVAL_DIR="./logs/${input}"
CHECKPOINT_PATH="./checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt"

echo "Dataset: ${DATASET_DIR}, log: ${EVAL_DIR}, ckpt: ${CHECKPOINT_PATH}"
mkdir -p "${EVAL_DIR}/demo"
rm -rf "${EVAL_DIR}/demo/"*

python demo_ssd_network.py \
    --eval_dir="${EVAL_DIR}" \
    --dataset_dir="${DATASET_DIR}" \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    "${@}"
