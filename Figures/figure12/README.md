
 
To denote the trend of the reduction of memory consumption and increase of training time as we increase the number of batches.  
we use the figure 12 (e) as an example: 1-layer GraphSAGE + LSTM aggregator  with different number of batches.  

After `./run_1-layer.sh` you can get the result of full batch, 2, 4, 8, 16, 32 micro batches.
in folder `log/`  
  
then run `data_collection.py` to collect the max memory consumption and average training time for different number of batches.  
you might get a table like below.
ogbn-products GraphSAGE fan-out 10 hidden 256

|    |   full batch  |      2 Micro batches |     4 Micro batches |     8 Micro batches |     16 Micro batches |    32   Micro batches |
|-------------------------------------------------|:--------------:|------------:|-------------:|------------:|-------------:|-------------:|
| average train time per epoch  (sec)                  |     0.0344584 |   0.0541086 |    0.110197 |     0.18126 |     0.285715 |     0.572999 |
| CUDA max memory consumption   (GB)                  |    13.0332    |   6.64646   |    3.3753   |     1.70534 |     0.87931  |     0.441572 |
| redundancy rate I (First Layer Input)           |     1         |   1.04004   |    1.12165  |     1.2015  |     1.27704  |     1.35091  |



