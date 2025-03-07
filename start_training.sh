#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --seed 59 \
    --dataset_name openvivqa \
    --epochs 40 \
    --patience 5 \
    --n_text_paras 2 \
    --text_para_thresh 0.8 \
    --n_text_para_pool 10 \
    --is_text_augment true \
    --use_dynamic_thresh true \
    --start_threshold 0.8 \
    --min_threshold 0.4 \
    --is_log_result true