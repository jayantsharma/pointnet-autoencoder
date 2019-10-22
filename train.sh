#!/bin/bash

python -u train.py \
  --max_epoch 15000 \
  --batch_size 32 \
  --model model_upconv \
  --epochs_to_wait=10000 \
  --surface_loss_wt=1e-1 \
  --learning_rate=1e-3 \
  --gpu 0 \
  --log_dir log_gae5layerbatch_lr1e-3_sl1e-1

# BN decay has been fixed to 1
