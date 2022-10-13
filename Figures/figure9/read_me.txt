## we pre-generated a full batch data and saved as .pickle file
#  Then we load the full batch data and micro-batch to check the distribution of in-degree correspondingly.
python full_Betty_arxiv_sage.py --num-batch 1 > log/full_batch_train.log
python full_Betty_arxiv_sage.py --num-batch 2 > log/2_micro_batch_train.log