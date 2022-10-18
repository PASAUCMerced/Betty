import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import sys

# sys.path.insert(0,'..')
# sys.path.insert(0,'../..')

def get_fan_out(filename):
	fan_out=filename.split('-')[3]
	# print(fan_out)
	return fan_out
def get_num_batch(filename):
	nb=filename.split('-')[9]
	# print(nb)
	return nb

def colored(r, g, b, text):
	return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def clear(infile):
	# print(infile)
	flag=True
	f = open(infile,'r')
	lst = []
	for line in f:
		if 'pytorch' in line or line.startswith('Using backend: pytorch'):
			line = line.replace('Using backend: pytorch',' ')
			flag=True
		lst.append(line)
	f.close()
	
	if len(lst) == 0:
		return [], False
	
	return lst, flag



def compute_efficiency_full_graph(filename):
	num_nid=[]
	num_output_nid=0
	full_graph_nid=0
	num_layer = 0   
	with open(filename) as f:
		train_times = []
		for line in f:
			line = line.strip()
			if line.startswith("NumTrainingSamples") or line.startswith("# Train:"):
				num_output_nid=int(line.split(':')[-1])

			if line.startswith("Training time/epoch"):
				train_times.append(float(line.split(' ')[-1]))
			if line.startswith("NumNodes:") or line.startswith("#nodes:") or line.startswith("# Nodes:"):
				full_graph_nid=int(line.split(' ')[-1])
			
			if line.startswith("The number of model layers:"):
				num_layer=int(line.split(' ')[-1])


	n_epoch=len(train_times)
	# print("num_output_nid*n_epoch")
	# print(num_output_nid*n_epoch)
	# print(train_times[-1])
	core_efficiency = (num_output_nid*n_epoch)/sum(train_times)
	
	# real_efficiency=0
	# if full_graph_nid:
	# print('full_graph_nid* num_layer*n_epoch')
	# print(full_graph_nid* num_layer*n_epoch)
	real_efficiency = (full_graph_nid * num_layer * n_epoch)/sum(train_times)
	return core_efficiency, real_efficiency, mean(train_times)


def get_first_layer_output_size(filename):
	first_layer_num_output_nid=[]
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith('Number of first layer output nodes during this epoch:'):
				first_layer_num_output_nid.append(int(line.split(' ')[-1])) 
			elif line.startswith('first layer output nodes number:'): # when OOM, the first layer output of full batch 
				first_layer_num_output_nid.append(int(line.split(' ')[-1])) 

	if len(first_layer_num_output_nid)==0:
		return 0
	return int(mean(first_layer_num_output_nid))




def get_full_batch_input_size(filename):
	first_layer_num_input_nid=[]
	
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith('Number of first layer input nodes during this epoch:'):
				first_layer_num_input_nid.append(int(line.split(' ')[-1])) 
			elif line.startswith('first layer input nodes number:'): # when OOM, the first layer output of full batch 
				first_layer_num_input_nid.append(int(line.split(' ')[-1])) 
			
	if len(first_layer_num_input_nid)==0:
		return 0
	return int(mean(first_layer_num_input_nid))


def compute_efficiency_full(filename, args,full_input_size=0):
	compute_num_nid=[]
	first_layer_num_input_nid=[]
	num_output_nid=0
	real_efficiency=0
	real_eff_wo_block_to_device=0
	redundancy =0
	sum_pseudo_input_size=0
	if full_input_size==0:
		redundancy=1
	epoch_times = []
	train_wo_to_device =[]
	pure_train_times=[]
	load_block_feature_label_times=[]
	block_to_device_times=[]

	core_pure_train=0
	real_pure_train=0
	OOM_flag=False
	
	# print(filename)
	f, flag = clear(filename)
	if flag:
	# with open(filename) as f:
		for line in f:
			line = line.strip()
			# print(line)
			if line.startswith("RuntimeError: CUDA out of memory."):
				OOM_flag=True
			if line.startswith("NumTrainingSamples") or line.startswith('# Train:'):
				num_output_nid=int(line.split(':')[-1])
			if line.startswith("Number of nodes for computation during this epoch:"):
				compute_num_nid.append(int(line.split(' ')[-1]))
			if line.startswith("Training time/epoch"):
				epoch_times.append(float(line.split(' ')[-1]))
			if line.startswith("Training time without block to device /epoch"):
				train_wo_to_device.append(float(line.split(' ')[-1]))
			if line.startswith("Training time without total dataloading part /epoch"):
				pure_train_times.append(float(line.split(' ')[-1]))
			
			if line.startswith("load block tensor time/epoch"):
				load_block_feature_label_times.append(float(line.split(' ')[-1]))
			if line.startswith("block to device time/epoch"):
				block_to_device_times.append(float(line.split(' ')[-1]))

			if line.startswith('Number of first layer input nodes during this epoch:'):
				first_layer_num_input_nid.append(float(line.split(' ')[-1]))


	if not OOM_flag:
		if len(train_wo_to_device)==0:
			# print(' there is no Training time without block to device /epoch !!!!!')
			return 0
		if len(compute_num_nid)!=len(train_wo_to_device) and len(train_wo_to_device)>0:
			# print('num_nid ', len(num_nid))
			# print('train_wo_to_device ', len(train_wo_to_device))
			nn_epoch=len(train_wo_to_device)
			compute_num_nid=compute_num_nid[-nn_epoch:]
			real_efficiency = sum(compute_num_nid)/sum(epoch_times)
			real_eff_wo_block_to_device = sum(compute_num_nid)/sum(train_wo_to_device) 
			core_eff_wo_block_to_device = (num_output_nid*nn_epoch)/sum(train_wo_to_device)
			real_pure_train=sum(compute_num_nid)/sum(pure_train_times)
			core_pure_train=(num_output_nid*nn_epoch)/sum(pure_train_times)
			
		
		n_epoch=len(epoch_times)
		core_efficiency = (num_output_nid*n_epoch)/sum(epoch_times)


	res={}
	if args.epoch_ComputeEfficiency and not OOM_flag:
		res.update({
			'final layer output nodes/epoch time':core_efficiency,
			'all layers input nodes/epoch time':real_efficiency, 
			# 'final layer output nodes/epoch time without block to device':core_eff_wo_block_to_device, 
			# 'all layers input nodes/epoch time without block to device':real_eff_wo_block_to_device,
			'average epoch time':mean(epoch_times)})
	if OOM_flag:		
		res.update({
			'final layer output nodes/pure train time':None, 
			'all layers input nodes//pure train time': None,
			
			# 'average epoch time w/o block to device':mean(train_wo_to_device),
			'average train time per epoch':None,
			# 'average dataloading time per epoch': (mean(epoch_times)-mean(pure_train_times)),
			'average number of nodes for computation':None,
			'average first layer num of input nodes':mean(first_layer_num_input_nid),
		})
	else:
		res.update({
				'final layer output nodes/pure train time':core_pure_train, 
				'all layers input nodes//pure train time': real_pure_train,
				
				# 'average epoch time w/o block to device':mean(train_wo_to_device),
				'average train time per epoch':mean(pure_train_times),
				# 'average dataloading time per epoch': (mean(epoch_times)-mean(pure_train_times)),
				'average number of nodes for computation':mean(compute_num_nid),
				'average first layer num of input nodes':mean(first_layer_num_input_nid),
		})

	if redundancy ==1:
		res.update({"redundancy rate (first layer input)":redundancy})

	if full_input_size >1:
		if not OOM_flag:
			res.update({"redundancy rate (first layer input)":mean(first_layer_num_input_nid)/full_input_size})
		else:
			res.update({"redundancy rate (first layer input)":None})

	if not OOM_flag:
		res.update({
			'average load block input feature time per epoch':mean(load_block_feature_label_times),
			'average block to device time per epoch':mean(block_to_device_times),
			'average dataloading time per epoch': mean(load_block_feature_label_times)+mean(block_to_device_times)
			})
	else:
		res.update({
				'average load block input feature time per epoch':None,
				'average block to device time per epoch':None,
				'average dataloading time per epoch': None
				})
	

	return res
	

def compute_efficiency(filename, args, full_input_size, first_layer_output_size, redundancy=0):
	compute_num_nid=[]
	first_layer_num_input_nid=[]
	first_layer_num_output_nid=[]
	num_output_nid=0
	real_efficiency=0
	real_eff_wo_block_to_device=0
	
	sum_pseudo_input_size=0
	
	epoch_times = []
	train_wo_to_device =[]
	pure_train_times=[]
	load_block_feature_label_times=[]
	block_to_device_times=[]

	core_pure_train=0
	real_pure_train=0
	OOM_flag=False
	cuda_max_mem=[]
	
	f, flag = clear(filename)
	if flag:
	# with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("RuntimeError: CUDA out of memory."):
				OOM_flag=True

			if line.startswith("NumTrainingSamples") or line.startswith('# Train:'):
				num_output_nid=int(line.split(':')[-1])
			if line.startswith("Number of nodes for computation during this epoch:"):
				compute_num_nid.append(int(line.split(' ')[-1]))
			if line.startswith("Training time/epoch"):
				epoch_times.append(float(line.split(' ')[-1]))
			if line.startswith("Training time without block to device /epoch"):
				train_wo_to_device.append(float(line.split(' ')[-1]))
			if line.startswith("Training time without total dataloading part /epoch"):
				pure_train_times.append(float(line.split(' ')[-1]))
			
			if line.startswith("load block tensor time/epoch"):
				load_block_feature_label_times.append(float(line.split(' ')[-1]))
			if line.startswith("block to device time/epoch"):
				if line.split(' ')[-1] == 'cuts':
					print('***********************************')
					print(filename)
				block_to_device_times.append(float(line.split(' ')[-1]))

			if line.startswith('Number of first layer input nodes during this epoch:'):
				if line.split(' ')[-1] == 'epoch:':
					print('***********************************')
					print(filename)
				first_layer_num_input_nid.append(float(line.split(' ')[-1]))
			if line.startswith('Number of first layer output nodes during this epoch:'):
				if line.split(' ')[-1] == 'epoch:':
					print('***********************************')
					print(filename)
				first_layer_num_output_nid.append(float(line.split(' ')[-1]))
			if line.startswith("Max Memory Allocated"):
				cuda_max_mem.append(float(line.split()[-2]))


	if not OOM_flag:
		if len(train_wo_to_device)==0:
			# print(' there is no Training time without block to device /epoch !!!!!')
			return 0
		if len(compute_num_nid)!=len(train_wo_to_device) and len(train_wo_to_device)>0:
			# print('num_nid ', len(num_nid))
			# print('train_wo_to_device ', len(train_wo_to_device))
			nn_epoch=len(train_wo_to_device)
			compute_num_nid=compute_num_nid[-nn_epoch:]
			real_efficiency = sum(compute_num_nid)/sum(epoch_times)
			real_eff_wo_block_to_device = sum(compute_num_nid)/sum(train_wo_to_device) 
			core_eff_wo_block_to_device = (num_output_nid*nn_epoch)/sum(train_wo_to_device)
			real_pure_train=sum(compute_num_nid)/sum(pure_train_times)
			core_pure_train=(num_output_nid*nn_epoch)/sum(pure_train_times)
			
		
		n_epoch=len(epoch_times)
		core_efficiency = (num_output_nid*n_epoch)/sum(epoch_times)


	res={}
	if args.epoch_ComputeEfficiency and not OOM_flag:
		res.update({
			# 'final layer output nodes/epoch time':core_efficiency,
			'all layers input nodes/epoch time':real_efficiency, 
			# 'final layer output nodes/epoch time without block to device':core_eff_wo_block_to_device, 
			# 'all layers input nodes/epoch time without block to device':real_eff_wo_block_to_device,
			'average epoch time':mean(epoch_times)})
	if OOM_flag:		
		res.update({
			# 'final layer output nodes/pure train time':None, 
			'all layers input nodes//pure train time': None,
			
			# 'average epoch time w/o block to device':mean(train_wo_to_device),
			'average train time per epoch':None,
			# 'average dataloading time per epoch': (mean(epoch_times)-mean(pure_train_times)),
			'average number of nodes for computation':None,
			'average first layer num of input nodes':None,
		})
	else:
		res.update({
				# 'final layer output nodes/pure train time':core_pure_train, 
				# 'all layers input nodes/pure train time': real_pure_train,
				
				# 'average epoch time w/o block to device':mean(train_wo_to_device),
				'average train time per epoch':mean(pure_train_times),
				# 'average dataloading time per epoch': (mean(epoch_times)-mean(pure_train_times)),
				# 'average number of nodes for computation':mean(compute_num_nid),
				# 'average first layer num of input nodes':mean(first_layer_num_input_nid),
				# 'average first layer num of output nodes':mean(first_layer_num_output_nid),
				'CUDA max memory consumption':max(cuda_max_mem),
		})

	if redundancy == 1:
		res.update({"redundancy rate I (First Layer Input)":1})
		# res.update({"redundancy rate O (First Layer Output)":1})
	if not OOM_flag:
		
		if full_input_size > 1 and first_layer_output_size > 1:
			res.update({"redundancy rate I (First Layer Input)":mean(first_layer_num_input_nid)/full_input_size})
			# res.update({"redundancy rate O (First Layer Output)":mean(first_layer_num_output_nid)/first_layer_output_size})
	
	if not OOM_flag:
		res.update({
			# 'average load block input feature time per epoch':mean(load_block_feature_label_times),
			# 'average block to device time per epoch':mean(block_to_device_times),
			# 'average dataloading time per epoch': mean(load_block_feature_label_times)+mean(block_to_device_times)
			})
	else:
		res.update({
				# 'average load block input feature time per epoch':None,
				# 'average block to device time per epoch':None,
				# 'average dataloading time per epoch': None
				})
	

	return res
	


				
def data_collection( path_2, args):
	
	# dict_full={}
	
	# fan_out=''
	
    nb_folder_list=[]

    for f_item in os.listdir(path_2):
        if 'batch-' in f_item:
            nb_size=f_item.split('-')[9]
            nb_folder_list.append(int(nb_size))
    nb_folder_list.sort()
    nb_folder_list=['1-layer-fo-10-sage-lstm-h-256-batch-'+str(i)+'-gp-REG.log' for i in nb_folder_list]

    res=[]
    full_batch_input_size=0
	
    full_batch_output_size_first_layer=0
    column_names=[]
    column_names_csv=[]
	
    for filename in nb_folder_list:
        
        f_ = os.path.join(path_2, filename)
        fan_out=get_fan_out(filename)
        nb=get_num_batch(filename)
        if int(nb) == 1:
            column_names.append('full batch \n fanout'+fan_out)
            column_names_csv.append('full batch fanout'+fan_out)
            full_batch_input_size=get_full_batch_input_size(f_)
            full_batch_output_size_first_layer=get_first_layer_output_size(f_)
            dict2 = compute_efficiency(f_, args, full_batch_input_size, full_batch_output_size_first_layer, redundancy=1)
        else:
            column_names.append('Micro \n'+str(nb)+' batches\n fanout'+fan_out)
            column_names_csv.append('Micro '+str(nb)+' batches fanout'+fan_out)
            dict2 = compute_efficiency(f_, args, full_batch_input_size, full_batch_output_size_first_layer)
        res += [dict2]
					
	
    df=pd.DataFrame(res).transpose()
    df.columns=column_names
    file_in = 'ogbn-products'
    model = 'GraphSAGE'
    df.index.name=file_in+' '+model +' fan-out '+fan_out+' hidden '+str(args.hidden)
    print(df.to_markdown(tablefmt="grid"))

    df_res=pd.DataFrame(res).transpose()
    df_res.columns=column_names_csv
    df_res.index.name=file_in + ' '+ args.model+' fan-out '+fan_out+' hidden '+str(args.hidden)
    save_ = args.save_path + 'log/'+args.selection_method+'/'+str(args.num_layers)+'_layer_'+args.file+'_'+args.aggre+'_'+str(args.hidden)
    df_res.to_csv(save_+'_eff.csv')


if __name__=='__main__':
	
	print("computation info data collection start ...... " )
	argparser = argparse.ArgumentParser("info collection")
	# argparser.add_argument('--file', type=str, default='cora')
	argparser.add_argument('--file', type=str, default='ogbn-products')
	# argparser.add_argument('--file', type=str, default='ogbn-arxiv')
	argparser.add_argument('--model', type=str, default='sage')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--hidden', type=int, default=32)
	argparser.add_argument('--hidden', type=int, default=256)
	# argparser.add_argument('--selection-method', type=str, default='range')
	# argparser.add_argument('--selection-method', type=str, default='random')
	argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--eval',type=bool, default=False)
	argparser.add_argument('--epoch-ComputeEfficiency', type=bool, default=False)
	argparser.add_argument('--epoch-PureTrainComputeEfficiency', type=bool, default=True)
	argparser.add_argument('--save-path',type=str, default='./')
	args = argparser.parse_args()
	
	path_2 = './log'
	data_collection( path_2, args)		





