We have pre-generated a full batch data and saved as a .pickle file.  
In micro batch train, we split the same full batch training data into two micro batches.  
Then we load the full batch data and micro-batch to check the distribution of in-degree correspondingly.  
`python full_Betty_arxiv_sage.py --num-batch 1 > log/full_batch_train.log`  
`python full_Betty_arxiv_sage.py --num-batch 2 > log/2_micro_batch_train.log`  

To save time, you can draw the figure 9 directly based on the log files in draw_figure folder.  
(We have generated the `full_batch_train.log` and `2_micro_batch_train.log`.)  
