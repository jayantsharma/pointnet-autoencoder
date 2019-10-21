#!/bin/bash

python -u train.py \
  --max_epoch 2000 \
  --batch_size 32 \
  --model model_upconv \
  --epochs_to_wait=0 \
  --surface_loss_wt=0 \
  --learning_rate=1e-3 \
  --gpu 0 \
  --log_dir log_chamfer_lr1e-3decayslow

# BN decay has been fixed to 1
