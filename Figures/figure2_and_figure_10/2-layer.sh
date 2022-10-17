#!/bin/bash


File=Betty_products_sage.py
# File=loading_data.py
Data=ogbn-products
# Data=ogbn-arxiv
# Data=cora
model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.01
dropout=0.5

run=1
epoch=5
logIndent=2

num_batch=(2 4 8 16 32 64)

pMethodList=(REG)

num_re_partition=(0)
# re_partition_method=REG
re_partition_method=random


layersList=(2)
fan_out_list=(10,25)

hiddenList=(256)
AggreList=(mean)

# savePath=./main_result/without_re/different_aggregator/1-epoch/arxiv/1-layer-REG/
# savePath=./main_result/without_re/different_aggregator/1-epoch/products/1-layer-REG/
savePath=./log/micro_batch_train/
# savePath=./dataloader_clean_folder/Betty-wo-duplicated/
for Aggre in ${AggreList[@]}
do      
	
	for pMethod in ${pMethodList[@]}
	do      
		
			for layers in ${layersList[@]}
			do      
				for hidden in ${hiddenList[@]}
				do
					for fan_out in ${fan_out_list[@]}
					do
						
						for nb in ${num_batch[@]}
						do
							
							for rep in ${num_re_partition[@]}
							do
								wf=${layers}-layer-fo-${fan_out}-sage-${Aggre}-h-${hidden}-batch-${nb}-gp-${pMethod}.log
								echo $wf

								python $File \
								--dataset $Data \
								--aggre $Aggre \
								--seed $seed \
								--setseed $setseed \
								--GPUmem $GPUmem \
								--selection-method $pMethod \
								--re-partition-method $re_partition_method \
								--num-re-partition $rep \
								--num-batch $nb \
								--lr $lr \
								--num-runs $run \
								--num-epochs $epoch \
								--num-layers $layers \
								--num-hidden $hidden \
								--dropout $dropout \
								--fan-out $fan_out \
								--log-indent $logIndent \
								--load-full-batch True \
								> ${savePath}${wf}

							done
						done
					done
				done
			done
		
	done
done
