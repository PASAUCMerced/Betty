import argparse
import dgl
import dgl.function as fn
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dgl.nn.pytorch import SAGEConv
from dgl.utils import expand_as_pair
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import tqdm
import sys
# sys.path.insert(0,'..')
from utils import Logger
from memory_usage import see_memory_usage, nvidia_smi_usage

from cpu_mem_usage import get_memory


class SAGEConv(nn.Module):
	def __init__(self,
				 in_feats,
				 out_feats,
				 aggregator_type,
				 bias=False
				 ):

		super(SAGEConv, self).__init__()

		self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
		self._out_feats = out_feats
		self._aggre_type = aggregator_type
		# aggregator type: mean/pool/lstm/gcn
		if aggregator_type == 'pool':
			self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
		if aggregator_type == 'lstm':
			self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
		if aggregator_type != 'gcn':
			self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
		# self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
		self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		# gain = nn.init.calculate_gain('relu')
		# nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
		# nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
		gain = nn.init.calculate_gain('relu')
		if self._aggre_type == 'pool':
			nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
		if self._aggre_type == 'lstm':
			self.lstm.reset_parameters()
		if self._aggre_type != 'gcn':
			nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
		nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
	
	def _lstm_reducer(self, nodes):
		"""LSTM reducer
		NOTE(zihao): lstm reducer with default schedule (degree bucketing)
		is slow, we could accelerate this with degree padding in the future.
		"""
		# print(nodes)
		m = nodes.mailbox['m'] # (B, L, D)
		print('m.shape '+str(m.shape))
		see_memory_usage("----------------------------------------1")
		batch_size = m.shape[0]
		see_memory_usage("----------------------------------------2")
		h = (m.new_zeros((1, batch_size, self._in_src_feats)),
			 m.new_zeros((1, batch_size, self._in_src_feats)))
		print(' h.shape '+ str(h[0].shape)+', '+ str(h[1].shape))
		see_memory_usage("----------------------------------------3")
		
		_, (rst, _) = self.lstm(m, h)
		see_memory_usage("----------------------------------------4")
		print('rst.shape ',rst.shape)
		return {'neigh': rst.squeeze(0)}



	def forward(self, graph, feat):
		r"""Compute GraphSAGE layer.
		Parameters
		----------
		graph : DGLGraph
			The graph.
		feat : torch.Tensor or pair of torch.Tensor
			If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
			:math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
			If a pair of torch.Tensor is given, the pair must contain two tensors of shape
			:math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
		Returns
		-------
		torch.Tensor
			The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
			is size of output feature.
		"""
		graph = graph.local_var()

		if isinstance(feat, tuple):
			feat_src, feat_dst = feat
		else:
			feat_src = feat_dst = feat
		if isinstance(feat, tuple):
			feat_src, feat_dst = feat
		else:
			feat_src = feat_dst = feat
			if graph.is_block:
				feat_dst = feat_src[:graph.number_of_dst_nodes()]
		
		msg_fn = fn.copy_src('h', 'm')
		h_self = feat_dst
		if self._aggre_type == 'mean':
			graph.srcdata['h'] =  feat_src
			graph.update_all(msg_fn, fn.mean('m', 'neigh'))
			h_neigh = graph.dstdata['neigh']
			h_neigh = self.fc_neigh(h_neigh)
		# graph.srcdata['h'] = feat_src
		# graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
		# h_neigh = graph.dstdata['neigh']
		elif self._aggre_type == 'pool':
			graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
			graph.update_all(msg_fn, fn.max('m', 'neigh'))
			h_neigh = self.fc_neigh(graph.dstdata['neigh'])
		elif self._aggre_type == 'lstm':
			graph.srcdata['h'] = feat_src
			see_memory_usage("----------------------------------------before graph.update_all(msg_fn, self._lstm_reducer)")
			graph.update_all(msg_fn, self._lstm_reducer)
			see_memory_usage("----------------------------------------after graph.update_all")
			
			h_neigh = self.fc_neigh(graph.dstdata['neigh'])
			see_memory_usage("----------------------------------------after h_neigh = self.fc_neigh")

		rst = self.fc_self(h_self) + h_neigh
		# see_memory_usage("----------------------------------------after rst")
		return rst



class GraphSAGE(nn.Module):
	def __init__(self,
				 in_feats,
				 hidden_feats,
				 out_feats,
				 aggre,
				 num_layers,
				 activation,
				 dropout):
		super(GraphSAGE, self).__init__()
		self.n_hidden = hidden_feats
		self.n_classes = out_feats
		self.activation = activation

		self.layers = nn.ModuleList()
		# self.bns = nn.ModuleList()
		if num_layers==1:
			self.layers.append(SAGEConv(in_feats, out_feats, aggre, bias=False))
		else:
			# input layer
			self.layers.append(SAGEConv(in_feats, hidden_feats, aggre, bias=False))
			# self.bns.append(nn.BatchNorm1d(hidden_feats))
			# hidden layers
			for _ in range(num_layers - 2):
				self.layers.append(SAGEConv(hidden_feats, hidden_feats, aggre, bias=False))
				# self.bns.append(nn.BatchNorm1d(hidden_feats))
			# output layer
			self.layers.append(SAGEConv(hidden_feats, out_feats, aggre, bias=False))
		self.dropout = nn.Dropout(p=dropout)

	def reset_parameters(self):
		for layer in self.layers:
			layer.reset_parameters()
		# for bn in self.bns:
		# 	bn.reset_parameters()

	def forward(self, blocks, x):
		for i, (layer, block) in enumerate(zip(self.layers[:-1], blocks[:-1])):
		# for i, layer in enumerate(self.layers[:-1]):
			
			# see_memory_usage("----------------------------------------before model layer "+str(i))
			# print(x.shape)
			x = layer(block, x)
			# see_memory_usage("----------------------------------------after model layer "+str(i)+ ' x = layer(block, x)')
			# print(x.shape)
			# if i==0:
			# 	print("first layer input nodes number: "+str(len(block.srcdata[dgl.NID])))
			# 	print("first layer output nodes number: "+str(len(block.dstdata[dgl.NID])))
			# else:
			# 	print("input nodes number: "+str(len(block.srcdata[dgl.NID])))
			# 	print("output nodes number: "+str(len(block.dstdata[dgl.NID])))
			# print("edges number: "+str(len(block.edges()[1])))
			# print("input nodes : "+str((blocks[-1].srcdata[dgl.NID])))
			# print("output nodes : "+str((blocks[-1].dstdata[dgl.NID])))
			# print("edges number: "+str((blocks[-1].edges())))
			# print("dgl.NID: "+str(dgl.NID))
			# print("dgl.EID: "+str((dgl.EID)))

			x = self.activation(x)
			# see_memory_usage("----------------------------------------after model layer "+str(i)+ " x = self.activation(x)")
			# print(x.shape)

			x = self.dropout(x)
			# see_memory_usage("----------------------------------------after model layer "+str(i)+' x = self.dropout(x)')
			# print(x.shape)

		x = self.layers[-1](blocks[-1], x)
		# see_memory_usage("----------------------------------------end of model layers  after x = self.layers[-1](blocks[-1], x)")
		# print(x.shape)
		# print("input nodes number: "+str(len(blocks[-1].srcdata[dgl.NID])))
		# print("output nodes number: "+str(len(blocks[-1].dstdata[dgl.NID])))
		# print("edges number: "+str(len(blocks[-1].edges()[1])))
		# print("input nodes : "+str((blocks[-1].srcdata[dgl.NID])))
		# print("output nodes : "+str((blocks[-1].dstdata[dgl.NID])))
		# print("edges number: "+str((blocks[-1].edges())))
		return x.log_softmax(dim=-1)

	def inference(self, g, x, args, device):
		"""
		Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
		g : the entire graph.
		x : the input of entire node set.

		The inference code is written in a fashion that it could handle any number of nodes and
		layers.
		"""
		# During inference with sampling, multi-layer blocks are very inefficient because
		# lots of computations in the first few layers are repeated.
		# Therefore, we compute the representation of all nodes layer by layer.  The nodes
		# on each layer are of course splitted in batches.
		# TODO: can we standardize this?
		device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
		for l, layer in enumerate(self.layers):
			y = torch.zeros(g.num_nodes(), self.n_hidden if l!=len(self.layers) - 1 else self.n_classes)

			sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
			dataloader = dgl.dataloading.NodeDataLoader(
				g,
				torch.arange(g.num_nodes(),dtype=torch.long).to(g.device),
				sampler,
				device=device,
				# batch_size=24,
				batch_size=args.batch_size,
				shuffle=True,
				drop_last=False,
				num_workers=args.num_workers)


			for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
				block = blocks[0]
				block = block.int().to(device)
				h = x[input_nodes].to(device)
				h = layer(block, h)
				# y[output_nodes] = h
				y[output_nodes] = h.cpu()

			x = y
		return y
