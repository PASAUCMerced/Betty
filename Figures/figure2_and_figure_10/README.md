## For different aggregator  
- Mini batch train  

To get the result of figure 2 (a)   
`./run_mini_batch_train_diff_aggregator.sh`   
You can find the max CUDA memory consumption    
Mean ggreagator:  8.44GB the last line in log/mini_batch_train/aggregators/2_layer_aggre_mean.log   
Pool ggreagator: 11.27GB the last line in log/mini_batch_train/aggregators/2_layer_aggre_pool.log  
Lstm aggregator: OOM  

- Micro batch train  

To get the result of figure 10 (a)   
Micro batch train can finish GraphSAGE model + lstm aggregator train without OOM.   
`./run_micro_batch_train_diff_aggregator.sh`  
When the full batch data split into 9 micro batch, it can break the memory wall.  
The max cuda memory consumption is about 21GB.   
more detail you can find in `log/micro_batch_train/2_layer_aggre_lstm_batch_9.log`

## For different layers  

### mini batch train  
- figure 2 (b)  
The result of mini batch train wiith different layers are located in `log/mini_batch_train/layers/`  
You can find the cuda memory consumption of different model layers (Graph SAGE model +mean aggreagtor)
1 layer: 7.66GB  
2 layer: 8.45GB  
3 layer: 13.25GB  Nvidia-smi: 19.01GB  
4 layer: 22.11GB  Nvidia-smi: 30.53GB when the GPU memory equals 24 GB, it will OOM.  
When the GPU memory capacity of the node is 32 GB, it can run 4-layer model successfully.  

- figure 10 (b)  
When GPU meomry constraint is 24 GB.  
The 4-layer will be OOM.  
To break the memory wall, Betty use 3 micro batches to train 4-layer model.
`./run_micro_batch_train_diff_layers.sh` it might takes a few minutes.
Then you will get the CUDA memory consumption of 4-layer model is 15.99GB. 


To save time, we only provide these two example to denote Our Betty breaks the memory capacity constraint in Figure 2.







