#!/bin/bash

python -u train.py \
  --model model_upconv \
  --gpu 0 \
  --log_dir log_gaefromstartsmall_lr1e-5
