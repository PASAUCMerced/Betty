#!/bin/bash
# echo "full batch train start..."
# python mini_batch_train.py --dataset ogbn-products \
#                         --aggre mean \
#                         --num-runs 1 \
#                         --num-epochs 500 \
#                         --num-hidden 32 \
#                         --num-layers 1 \
#                         --fan-out 5 \
#                         --batch-size 196571 \
#                         --lr 0.003 \
#                         --dropout 0.5 \
#                         --weight-decay 5e-4 \
#                         --eval \
#                         > log/full_batch_train_hidden_32_layer_1.log

# python mini_batch_train.py --dataset ogbn-products \
#                         --aggre mean \
#                         --num-runs 1 \
#                         --num-epochs 500 \
#                         --num-hidden 32 \
#                         --num-layers 3 \
#                         --fan-out 5,10,15 \
#                         --batch-size 196571 \
#                         --lr 0.0001 \
#                         --dropout 0.5 \
#                         --weight-decay 5e-4 \
#                         --eval \
#                         > log/full_batch_train_hidden_32_layer_3.log
# python mini_batch_train.py --dataset ogbn-products \
#                         --aggre mean \
#                         --num-runs 1 \
#                         --num-epochs 500 \
#                         --num-hidden 32 \
#                         --num-layers 1 \
#                         --fan-out 5 \
#                         --batch-size 12289 \
#                         --lr 0.0001 \
#                         --dropout 0.5 \
#                         --weight-decay 5e-6 \
#                         --eval \
#                         > log/mini_batch_train_hidden_32.log
# python mini_batch_train.py --dataset ogbn-products \
#                         --aggre mean \
#                         --num-runs 1 \
#                         --num-epochs 500 \
#                         --num-hidden 32 \
#                         --num-layers 3 \
#                         --fan-out 5,10,15 \
#                         --batch-size 12289 \
#                         --lr 0.003 \
#                         --dropout 0.5 \
#                         --weight-decay 5e-4 \
#                         --eval \
#                         > log/mini_batch_train_hidden_32_lr_0.003.log
echo "mini batch train start..."
python mini_batch_train.py --dataset ogbn-products \
                        --aggre mean \
                        --num-runs 1 \
                        --num-epochs 500 \
                        --num-hidden 32 \
                        --num-layers 3 \
                        --fan-out 5,10,15 \
                        --num-batch 16 \
                        --lr 0.005 \
                        --dropout 0.5 \
                        --weight-decay 5e-4 \
                        --eval \
                        > log/mini_batch_train_hidden_32_lr_0.005.log