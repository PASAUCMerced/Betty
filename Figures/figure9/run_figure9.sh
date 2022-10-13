#!/bin/bash
python full_Betty_arxiv_sage.py --num-batch 1 > log/full_batch_train.log
python full_Betty_arxiv_sage.py --num-batch 2 > log/2_micro_batch_train.log