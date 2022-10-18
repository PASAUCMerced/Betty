#!/bin/bash

mkdir ./log1
mkdir ./log1/mini_batch_train
save_path=./log1/mini_batch_train/aggregators
mkdir  $save_path

echo '---start mini batch train: mean aggregator. It only trains 5 epoch to save time, it will last about 1 min.'
python mini_batch_train.py \
    --dataset ogbn-products \
    --num-batch 1 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden 256 \
    --num-runs 1 \
    --num-epoch 5 \
    --eval \
    --aggre mean \
    > ${save_path}/2_layer_aggre_mean.log

echo '---start mini batch train: pool aggregator. It only trains 5 epoch to save time, it will last about 1 min.'
python mini_batch_train.py \
    --dataset ogbn-products \
    --num-batch 1 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden 256 \
    --num-runs 1 \
    --num-epoch 5 \
    --eval \
    --aggre pool \
    > ${save_path}/2_layer_aggre_pool.log

echo '---start mini batch train: lstm aggregator. It only trains 5 epoch to save time.'
python mini_batch_train.py \
    --dataset ogbn-products \
    --num-batch 1 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden 256 \
    --num-runs 1 \
    --num-epoch 5 \
    --eval \
    --aggre lstm \
    > ${save_path}/2_layer_aggre_lstm.log
