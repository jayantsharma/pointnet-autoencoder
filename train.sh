#!/bin/bash

python -u train.py \
  --max_epoch 100 \
  --batch_size 32 \
  --model model_upconv \
  --gpu 1 \
  --learning_rate=1e-4 \
  --epochs_to_wait=100 \
  --surface_loss_wt=1e-1 \
  --log_dir log_bn
