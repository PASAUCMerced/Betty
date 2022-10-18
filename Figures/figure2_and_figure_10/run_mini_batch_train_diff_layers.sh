#!/bin/bash

mkdir ./log1
mkdir ./log1/mini_batch_train
save_path=./log1/mini_batch_train/layers
mkdir $save_path
echo '---start mini batch train: 1-layer. It only trains 5 epoch to save time, it will last about 1 min.'
python mini_batch_train.py \
    --dataset ogbn-products \
    --num-batch 1 \
    --num-layers 1 \
    --fan-out 10 \
    --num-hidden 256 \
    --num-runs 1 \
    --num-epoch 5 \
    --eval \
    --aggre mean \
    > ${save_path}/1_layer_aggre_mean.log

echo '---start mini batch train: 2-layers. It only trains 5 epoch to save time, it will last about 1 min.'
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

echo '---start mini batch train: 3-layers. It only trains 5 epoch to save time.'
python mini_batch_train.py \
    --dataset ogbn-products \
    --num-batch 1 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 256 \
    --num-runs 1 \
    --num-epoch 5 \
    --eval \
    --aggre mean \
    > ${save_path}/3_layer_aggre_mean.log
echo '---start mini batch train: 4-layers. It only trains 5 epoch to save time.'
python mini_batch_train.py \
    --dataset ogbn-products \
    --num-batch 1 \
    --num-layers 4 \
    --fan-out 10,25,30,40 \
    --num-hidden 256 \
    --num-runs 1 \
    --num-epoch 5 \
    --eval \
    --aggre mean \
    > ${save_path}/4_layer_aggre_mean.log
