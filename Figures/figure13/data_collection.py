import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

	
def read_acc_loss(filename):
    acc_array = []
    loss_array = []
    with open(filename) as f:
        for line in f:
            if ('Run'in line.strip() )and ( 'Test' in line.strip()):
                # print(type(acc))
                acc=line.split()[-1]
                loss=line.split()[7]
                acc = float(acc)
                loss = float(loss)
                acc_array.append(acc)
                loss_array.append(loss)
                
    print(acc_array[:10])
    print(len(acc_array))
    return acc_array

def data_formalize(acc_list):
    len_1=len(acc_list[0])
    len_2=len(acc_list[1])
    len_4=len(acc_list[2])
    len_8=len(acc_list[3])
    len_cut = min([len_1,len_2, len_4, len_8 ])

    for i in range(4):
        acc_list[i] = acc_list[i][:len_cut]
    
    x=range(len_cut)

    return x, acc_list[0], acc_list[1], acc_list[2], acc_list[3]

def draw(acc_list):
	fig, ax = plt.subplots( figsize=(7, 4))
	# ax.set_facecolor('0.8')
    
	x, acc_1, acc_2, acc_4, acc_8,= data_formalize(acc_list)
	ax.plot(x, acc_1, '-',label='full batch train', color='orange')
	ax.plot(x, acc_2, '--',label='2 micro batch train', color='purple')
	ax.plot(x, acc_4, '-.',label='4 micro batch train', color='lime')
	ax.plot(x, acc_8, ':', label='8 micro batch train', color='red')
	
	
	ax.set(xlabel='Epoch', ylabel='Test Accuracy')
	plt.legend()
	plt.savefig('Figure13.png')
	

				
def data_collection( path, args):
	
	nb_folder_list=[]
	acc_list = []

	for f_item in os.listdir(path):
		if 'batch-' in f_item:
			nb_size=f_item.split('-')[9]
			nb_folder_list.append(int(nb_size))
	nb_folder_list.sort()
	nb_folder_list=['1-layer-fo-10-sage-mean-h-16-batch-'+str(i)+'-gp-REG.log' for i in nb_folder_list]
	for file in nb_folder_list:
		acc_list.append( read_acc_loss(path+file) )
		
	draw(acc_list )


if __name__=='__main__':
    
	print("computation info data collection start ...... " )
	argparser = argparse.ArgumentParser("info collection")
	# argparser.add_argument('--file', type=str, default='cora')
	# argparser.add_argument('--file', type=str, default='ogbn-products')
	argparser.add_argument('--file', type=str, default='ogbn-arxiv')
	argparser.add_argument('--model', type=str, default='sage')
	argparser.add_argument('--aggre', type=str, default='mean')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--hidden', type=int, default=32)
	argparser.add_argument('--hidden', type=int, default=16)
	# argparser.add_argument('--selection-method', type=str, default='range')
	# argparser.add_argument('--selection-method', type=str, default='random')
	argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--eval',type=bool, default=False)
	argparser.add_argument('--epoch-ComputeEfficiency', type=bool, default=False)
	argparser.add_argument('--epoch-PureTrainComputeEfficiency', type=bool, default=True)
	argparser.add_argument('--save-path',type=str, default='./')
	args = argparser.parse_args()
	
	path = './log/'
	data_collection( path, args)		





