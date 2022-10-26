# Betty: Enabling Large-Scale GNN Training with Batch-Level Graph Partitioning  

#### Well-prepared nodes on Chameleon cloud for artifact evaluation:   
**ssh cc@192.5.86.155, password: test**   (terminated, please try another two)    
**ssh cc@192.5.87.23, password: test**  
**ssh cc@192.5.86.188, password: test**   
As the cloud node might be reserved by others, the IP address might be different later, please check this file when you ssh access the cloud node.
If you have any questions, you can email me directly.  
Email: syang127@ucmerced.edu


## install requirements:
 The framework of Betty is developed upon DGL(pytorch backend)  
 We use Ubuntu 18.04, CUDA 11.2 (it's also compatible with Ubuntu16.04, CUDA 10.1, the package version you need to install are denoted in install_requirements.sh).  
 The requirements:  pytorch >= 1.7, DGL >= 0.7

`bash install_requirements.sh`.  

## Our main contributions: 
Betty introduces two novel techniques, redundancy-embedded graph (REG) partitioning and memory-aware partitioning, to effectively mitigate the redundancy and load imbalances issues across the partitions. 


- redundancy-embedded graph (REG) is implemented in  
```python
~/Betty/pytorch/micro_batch_train/graph_partitioner.py  
```
- memory-aware partitioning implementation is based on memory estimation, details are in  
```python 
~/Betty/pytorch/micro_batch_train/block_dataloader.py  
```




### The main steps for code reproduction on your own device:  
- step0:   
    `git clone https://github.com/HaibaraAiChan/Betty.git`. 
- step1: generate some full batch data for later experiments, (the generated data will be stored in ~/Betty/dataset/).
    `cd /Betty/pytorch/micro_batch_train/`  
   `./gen_data.sh`    
- step2:   
    `cd Figures/figureXXX/` to test the experiments follow the instruction in `README.md` in corresponding figure folder.  
    And the expected results are in bak folder in each figure folder.  
   






