#!/bin/bash


File=Betty_products_sage.py
Data=ogbn-products

model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.01
dropout=0.5

run=1
epoch=1
logIndent=0

pMethodList=(REG)

re_partition_method=REG
re_partition_method=random
# pMethodList=( random range) 

num_batch=(3)
num_re_partition=(0)

layersList=(4)
fan_out_list=(10,25,30,40)

hiddenList=(256 )
AggreList=(mean )


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
							echo 'number of batches equals '${nb}
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
							> ./log/micro_batch_train/layers/${layers}_layer_aggre_${Aggre}_batch_${nb}.log

							done
						done
					done
				done
			done
		
	done
done

# num_batch=(60)
# layersList=(5)
# fan_out_list=(10,25,30,40,50)
# for Aggre in ${AggreList[@]}
# do      
	
# 	for pMethod in ${pMethodList[@]}
# 	do      
		
# 			for layers in ${layersList[@]}
# 			do      
# 				for hidden in ${hiddenList[@]}
# 				do
# 					for fan_out in ${fan_out_list[@]}
# 					do
						
# 						for nb in ${num_batch[@]}
# 						do
							
# 							for rep in ${num_re_partition[@]}
# 							do
# 							echo 'number of batches equals '${nb}
# 							python $File \
# 							--dataset $Data \
# 							--aggre $Aggre \
# 							--seed $seed \
# 							--setseed $setseed \
# 							--GPUmem $GPUmem \
# 							--selection-method $pMethod \
# 							--re-partition-method $re_partition_method \
# 							--num-re-partition $rep \
# 							--num-batch $nb \
# 							--lr $lr \
# 							--num-runs $run \
# 							--num-epochs $epoch \
# 							--num-layers $layers \
# 							--num-hidden $hidden \
# 							--dropout $dropout \
# 							--fan-out $fan_out \
# 							--log-indent $logIndent \
# 							--load-full-batch True \
# 							> ./log/micro_batch_train/layers/${layers}_layer_aggre_${Aggre}_batch_${nb}.log

# 							done
# 						done
# 					done
# 				done
# 			done
		
# 	done
# done
