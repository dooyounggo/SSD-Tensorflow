#!/bin/bash

echo "Enter directory name:"
read input

DATASET_DIR="./tfrecords/${input}"
EVAL_DIR="./logs/${input}"
CHECKPOINT_PATH="./checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt"

echo "Dataset: ${DATASET_DIR}, log: ${EVAL_DIR}, ckpt: ${CHECKPOINT_PATH}"
mkdir -p ${EVAL_DIR}
rm -rf ${EVAL_DIR}/events*

python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    ${@}

