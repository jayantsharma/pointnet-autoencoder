#!/bin/bash

LOG_DIR=log_gae5layerbatch_postinit_lr1e-4_sl5e-1

# nohup tensorboard --logdir=$LOG_DIR --port=6022 &> /dev/null &

nohup python -u train.py \
  --max_epoch 20000 \
  --batch_size 32 \
  --model model_upconv \
  --epochs_to_wait=10000 \
  --surface_loss_wt=5e-1 \
  --learning_rate=1e-4 \
  --gpu 1 \
  --log_dir $LOG_DIR \
  &> foo.log &

# BN decay has been fixed to 1
