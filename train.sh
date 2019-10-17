#!/bin/bash

python -u train.py \
  --max_epoch 50 \
  --batch_size 32 \
  --model model_upconv \
  --epochs_to_wait=0 \
  --surface_loss_wt=1 \
  --learning_rate=1e-6 \
  --gpu 1 \
  --log_dir log_gaeonly_lr1e-6

# BN decay has been fixed to 1
