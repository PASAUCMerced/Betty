import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../pytorch/utils/')
sys.path.insert(0,'../../pytorch/models/')
import time
import argparse
import math
import numpy as np
import random
import torch
import dgl

import pickle
from memory_usage import see_memory_usage
from load_graph import load_ogbn_dataset
from collections import Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
	'Counter that remembers the order elements are first encountered'

	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

	def __reduce__(self):
		return self.__class__, (OrderedDict(self),)


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)

def run(args, device, data):
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	if args.num_batch == 1:
		args.batch_size = len(train_nid)
	if args.batch_size == 0:
		if len(train_nid)%args.num_batch==0:
			args.batch_size = len(train_nid)//args.num_batch
		else:
			args.batch_size = len(train_nid)//args.num_batch + 1
	full_batch_dataloader = dgl.dataloading.NodeDataLoader(
		g,
		train_nid,
		sampler,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=0)
	if args.load_full_batch:
		full_batch_dataloader=[]
		file_name=r'./../../dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_0_items.pickle'
		with open(file_name, 'rb') as handle:
			item=pickle.load(handle)
			full_batch_dataloader.append(item)
	for step, (input_nodes, seeds, blocks) in enumerate(full_batch_dataloader):
		degrees = blocks[0].in_degrees()
		distribution =  dict(Counter(degrees.tolist()))
		distribution = dict(OrderedDict(sorted(distribution.items())))
		print(distribution)
        


def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=1,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--batch-size', type=int, default=0) 
	argparser.add_argument('--num-epochs', type=int, default=1)

	argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--num-batch', type=int, default=9)

	argparser.add_argument('--num-hidden', type=int, default=256)

	argparser.add_argument('--num-layers', type=int, default=1)
	argparser.add_argument('--fan-out', type=str, default='10')
	
	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)
	
	args = argparser.parse_args()
	if args.setseed:
		set_seed(args)
	device = "cpu"
	if args.GPUmem:
		see_memory_usage("-----------------------------------------before load data ")
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='pubmed':
		g, n_classes = load_pubmed()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset == 'ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		device = "cuda:0"

	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
	else:
		raise Exception('unknown dataset')
		
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	main()

