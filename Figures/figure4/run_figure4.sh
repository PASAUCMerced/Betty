#!/bin/bash

python mini_batch_train.py --dataset ogbn-products \
                        --aggre mean \
                        --num-runs 1 \
                        --num-epochs 500 \
                        --num-hidden 32 \
                        --num-layers 3 \
                        --fan-out 5,10,15 \
                        --batch-size 196571 \
                        --lr 0.003 \
                        --dropout 0.5 \
                        --weight-decay 5e-4 \
                        --eval \
                        > log/full_batch_train_hidden_32.log

python mini_batch_train.py --dataset ogbn-products \
                        --aggre mean \
                        --num-runs 1 \
                        --num-epochs 500 \
                        --num-hidden 32 \
                        --num-layers 3 \
                        --fan-out 5,10,15 \
                        --batch-size 16 \
                        --lr 0.003 \
                        --dropout 0.5 \
                        --weight-decay 5e-4 \
                        --eval \
                        > log/mini_batch_train_hidden_32.log