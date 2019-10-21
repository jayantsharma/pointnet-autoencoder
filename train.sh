#!/bin/bash

python -u train.py \
  --max_epoch 200 \
  --model model_upconv \
  --epochs_to_wait=100 \
  --surface_loss_wt=5e-1 \
  --gpu 0 \
  --learning_rate=1e-5 \
  --log_dir log_gae5layersmall_lr1e-5_sl5e-1

# BN decay has been fixed to 1
