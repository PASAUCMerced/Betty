## Betty: Enabling Large-Scale GNN Training with Batch-Level Graph Partitioning  

 The framework of Betty is developed upon DGL(pytorch backend)  
 the requirements:  pytorch >= 1.7, DGL >= 0.7



  Well-prepared node on Chameleon cloud: **ssh cc@192.5.86.155, password: test**



Betty introduces two novel techniques, redundancy-embedded graph (REG) partitioning and memory-aware partitioning, to effectively mitigate the redundancy and load imbalances issues across the partitions. 


redundancy-embedded graph (REG)is implemented in  
```python
micro_batch_train/graph_partitioner.py  
```
memory-aware partitioning implementation is based on memory estimation, details are in  
```python 
micro_batch_train/block_dataloader.py  
```








