## Betty: Enabling Large-Scale GNN Training with Batch-Level Graph Partitioning  

#### Well-prepared node on Chameleon cloud for artifact evaluation:   
**ssh cc@192.5.86.155, password: test**



## install requirements
 The framework of Betty is developed upon DGL(pytorch backend)  
 We use Ubuntu 18.04, CUDA 11.2 (it's also compatible with Ubuntu16.04, CUDA 10.1).  
 The requirements:  pytorch >= 1.7, DGL >= 0.7

`pip -r install requirements.txt`.  

## Our main contributions. 
Betty introduces two novel techniques, redundancy-embedded graph (REG) partitioning and memory-aware partitioning, to effectively mitigate the redundancy and load imbalances issues across the partitions. 


- redundancy-embedded graph (REG) is implemented in  
```python
micro_batch_train/graph_partitioner.py  
```
- memory-aware partitioning implementation is based on memory estimation, details are in  
```python 
micro_batch_train/block_dataloader.py  
```




### The main steps for code reproduction on your own device.  
- step1: generate some full batch data for later experiments, (the generated data will be stored in ~/Betty/dataset/).
   `./~Betty/pytorch/micro_batch_train/gen_data.sh`.   
- step2:   
    `cd Figures/figureXXX/` to test the experiments follow the instruction in `README.md` in corresponding figure folder.  
    And the expected results are in bak folder in each figure folder.  
   






