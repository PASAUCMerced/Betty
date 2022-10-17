## Mini batch train

To get the result of figure 2 (a)  
`./run_mini_batch_train.sh`  
You can find the max CUDA memory consumption   
Mean ggreagator:  8.44GB the last line in log/run_mini_batch_train/2_layer_aggre_mean.log  
Pool ggreagator: 11.27GB the last line in log/run_mini_batch_train/2_layer_aggre_pool.log  
Lstm aggregator: OOM

## Micro batch train

To get the result of figure 10 (a)  
Micro batch train can finish GraphSAGE model + lstm aggregator train without OOM.  
`./run_micro_batch_train_a.sh`

