#!/bin/bash

python -u train.py \
  --max_epoch 15000 \
  --batch_size 32 \
  --model model_upconv \
  --epochs_to_wait=10000 \
  --surface_loss_wt=1 \
  --learning_rate=1e-4 \
  --gpu 1 \
  --log_dir log_gae5layerbatch_lr1e-4

# BN decay has been fixed to 1
