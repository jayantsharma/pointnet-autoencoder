#!/bin/bash

python -u train.py \
  --max_epoch 20 \
  --model model_upconv \
  --gpu 0 \
  --learning_rate=1e-4 \
  --surface_loss_wt=0 \
  --log_dir log_gae_baseline
