#!/bin/bash

python -u train.py \
  --max_epoch 72 \
  --batch_size 32 \
  --model model_upconv \
  --learning_rate=1e-3 \
  --epochs_to_wait=0 \
  --gpu 0 \
  --surface_loss_wt=1 \
  --log_dir log_baz
