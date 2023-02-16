# Betty: Enabling Large-Scale GNN Training with Batch-Level Graph Partitioning 
#### [paper link [pdf]](https://dl.acm.org/doi/pdf/10.1145/3575693.3575725)



## Install requirements:
 The framework of Betty is developed upon DGL(pytorch backend)  
 We use Ubuntu 18.04, CUDA 11.2,   
   
 (it's also compatible with Ubuntu16.04, CUDA 10.1, the package version you need to install are denoted in install_requirements.sh).  
 The requirements:  pytorch >= 1.7, DGL >= 0.7  
 
 (python 3.6 is the basic configuration in requirements here, you can use other python version, e.g. python3.8, you need configure the corresponding pytorch and dgl version.)  

`bash install_requirements.sh`. 


## Structure of dirctlory  
- The directory **/pytorch** contains all necessary files for the micro-batch training and mini-batch training.   
In folder micro_batch_train, `graph_partitioner.py` contains our implementation of redundancy embedded graph partitioning.
`block_dataloader.py` is implemented to construct the micro-batch based on the partitioning results of REG.  
- You can download the benchmarks and generate full batch data into folder **/dataset**.  
- The folder **/Figures** contains these important figures for analysis and performance evaluation.


### The main steps for code reproduction on your own device:  
- step0: Obtain the artifact, extract the archive files     
    `git clone https://github.com/HaibaraAiChan/Betty.git`. 
- step1: generate some full batch data for later experiments, (the generated data will be stored in ~/Betty/dataset/).  
    `cd /Betty/pytorch/micro_batch_train/`    
   `./gen_data.sh`    
- step2: replicate these experiments in **Figures/**    
    `cd Figures/figureXXX/` to test the experiments follow the instruction in `README.md` in corresponding figure folder.    
    And the expected results are in bak folder in each figure folder.    
   






