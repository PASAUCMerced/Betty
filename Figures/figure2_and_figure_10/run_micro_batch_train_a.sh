#!/bin/bash

echo '---start mini batch train: lstm aggregator. \
It only trains 5 epoch to save time, it will last about 1 min.'
python Betty_micro_batch_train.py \
    --device 1\
    --dataset ogbn-products \
    --num-batch 9 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden 256 \
    --num-runs 1 \
    --num-epoch 5 \
    --eval \
    --aggre lstm \
    > log/micro_batch_train/2_layer_aggre_lstm.log


