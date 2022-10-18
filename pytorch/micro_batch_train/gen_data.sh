#!/bin/bash

File=gen_data.py

# data=ogbn-products
# data=cora
# data=pubmed
# data=reddit

data=ogbn-arxiv
fan_out=10
mkdir ~/Betty/dataset/fan_out_10
python $File --fan-out=$fan_out --num-layers=1 --num-epochs=180 --num-hidden=1 --dataset=$data

num_epoch=5

fan_out=10,25
mkdir ~/Betty/dataset/fan_out_10,25
python $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
#--------------------------------------------------------------------------------------------------------
data=ogbn-products

fan_out=10
python $File --fan-out=$fan_out --num-layers=1 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data


fan_out=10,25
python $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data


fan_out=10,25,30
mkdir ~/Betty/dataset/fan_out_10,25,30
python $File --fan-out=$fan_out --num-layers=3 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data


fan_out=10,25,30,40
mkdir ~/Betty/dataset/fan_out_10,25,30,40
python $File --fan-out=$fan_out --num-layers=4 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data


fan_out=10,25,30,40,50
mkdir ~/Betty/dataset/fan_out_10,25,30,40,50
python $File --fan-out=$fan_out --num-layers=5 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data


