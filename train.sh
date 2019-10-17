#!/bin/bash

python -u train.py \
  --max_epoch 50 \
  --model model_upconv \
  --epochs_to_wait=0 \
  --surface_loss_wt=1 \
  --gpu 0 \
  --learning_rate=1e-6 \
  --log_dir log_gaeonlysmall_lr1e-6

# BN decay has been fixed to 1
