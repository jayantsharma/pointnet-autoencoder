#!/bin/bash

python -u train.py \
  --model model_upconv \
  --gpu 0 \
  --log_dir log_gae_baseline
