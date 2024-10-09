# -*- coding: utf-8 -*-
import csv
import logging
import random
import time
from collections import defaultdict
from random import Random
import torch

import numpy as np
import os
from torch.utils.data import DataLoader
import fedscale.core.config_parser as parser

import PIL.Image as Image
import io
from io import BytesIO

import redis
import heapdict
import PIL
from rediscluster import RedisCluster
from collections import OrderedDict
import ast
import math
#from argParser import args


class Partition(object):
	""" Dataset partitioning helper """

	def __init__(self, data, index):
		self.data = data
		self.index = index

	def __len__(self):
		return len(self.index)

	def __getitem__(self, index):
		data_idx = self.index[index]
		return self.data[data_idx]


class DataPartitioner(object):
	"""Partition data by trace or random"""

	def __init__(self, data, args, numOfClass=0, seed=10, isTest=False):
		self.partitions = []
		self.rng = Random()
		self.rng.seed(seed)

		self.data = data
		self.labels = self.data.targets
		self.args = args
		self.isTest = isTest
		np.random.seed(seed)

		self.data_len = len(self.data)
		self.task = args.task
		self.numOfLabels = numOfClass
		self.client_label_cnt = defaultdict(set)
		self.checker = {}

		self.startup_nodes = [{"host": self.args.host_ip, "port": self.args.port_num}]
		self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)

	def getNumOfLabels(self):
		return self.numOfLabels

	def getDataLen(self):
		return self.data_len

	def getClientLen(self):
		return len(self.partitions)

	def getClientLabel(self):
		return [len(self.client_label_cnt[i]) for i in range(self.getClientLen())]

	def read_imp_mapping_from_csv(self, file_path):
		samptoimp_map = {}
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)  # Skip header 
			for row in reader:
				samp = int(row[0])
				imp = float(row[1])
				samptoimp_map[samp] = imp
		return samptoimp_map

	def read_imp_freq_mapping_from_csv(self, file_path):
		samptoimpfreq_map = {}
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)  # Skip header
			for row in reader:
				samp = int(row[0])
				tup = ast.literal_eval(row[1])
				imp = float(tup[0])
				freq = float(tup[1])
				samptoimpfreq_map[samp] = (imp,freq)
			return samptoimpfreq_map


	def read_param(self,file_path):
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)  # Skip header 
			for row in reader:
				prev_loss = float(row[0])
				ls_param = float(row[1])
				#samptoimp_map[samp] = imp
		return prev_loss, ls_param

	def save_imp_mapping_to_csv(self, l1, l2, file_path):
		mapping = dict(zip(l1, l2))
		with open(file_path, 'w', newline='') as csvfile:
			#logging.info(f"file is opened {file_path}")
			writer = csv.writer(csvfile)
			#logging.info(f"csv writer initialized.")
			writer.writerow(['sample', 'imp'])  # Write header
			#logging.info(f"header row initialized.")
			writer.writerows(mapping.items())
			#logging.info(f"items mapped in filepath.")

	def save_imp_freq_mapping_to_csv(self, l1, l2, l3, file_path):
		mapping = {key: (val1, val2) for key, val1, val2 in zip(l1, l2, l3)}

		with open(file_path, 'w', newline='') as csvfile:
			#logging.info(f"file is opened {file_path}")
			writer = csv.writer(csvfile)
			#logging.info(f"csv writer initialized.")
			writer.writerow(['sample', 'imp', 'freq'])  # Write header
			#logging.info(f"header row initialized.")
			writer.writerows(mapping.items())
			#logging.info(f"items mapped in filepath.")

	def save_client_mapping(self, unique_clientIds, file_path):

		#logging.info(f"save_client_mapping called {unique_clientIds}")
		with open(file_path, 'w', newline='') as csvfile:
			#logging.info(f"file is opened {file_path}")
			writer = csv.writer(csvfile)
			#logging.info(f"csv writer initialized.")
			writer.writerow(['real', 'assigned'])  # Write header
			#logging.info(f"header row initialized.")
			writer.writerows(unique_clientIds.items())

	def write_client_numcalls(self,clientId, count, file_path):

		with open(file_path, 'w', newline = '') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['clientId', 'count'])
			writer.writerow([clientId, count])


	def read_client_numcalls(self,file_path):
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)  # Skip header 
			for row in reader:
				clientId = str(row[0])
				count = float(row[1])
				#samptoimp_map[samp] = imp
		return count


	def pads_sort(self,l):
		# l -> list to be sorted
		n = len(l)

		# d is a hashmap
		d = {}
		#d = defaultdict(lambda: 0)
		for i in range(n):
			#d[l[i]] += 1
			d[l[i]] = 1 + d.get(l[i],0)

		# Sorting the list 'l' where key
		# is the function based on which
		# the array is sorted
		# While sorting we want to give
		# first priority to Frequency
		# Then to value of item
		l.sort(key=lambda x: (-d[x], x))

		return l

	def prepare_hits(self,partition, indices, needtotake, r):
		hit_list = []
		miss_list = []
		prev_r = r
		#total_samples_needed = len(indices)
		total_samples_needed = needtotake
		for ind in indices:
			key = partition + "_" + str(ind)
			if self.key_id_map.exists(key):
				hit_list.append(ind)
			else:
				miss_list.append(ind)

		samples_in_cache = len(hit_list)
		if samples_in_cache != 0:
			mul_factor = (total_samples_needed // samples_in_cache) + 1
		else:
			mul_factor = r

		if mul_factor <= 3:
			r = mul_factor
		elif samples_in_cache > int(self.args.batch_size)*0.6: #at least 60% samples is not repetitions in each batch
			mul_factor = mul_factor * 0.5
			r = math.ceil(mul_factor)
			if r < 3: r = 3
		else:
			r = int(self.args.batch_size)*0.2 #make sure at least 80% of each batch has different samples
			if r < 3: r = 3

		if self.args.subs == 1.0 or self.args.base_case:
			r = prev_r
		print(f'rank:{partition} hit_list_len: {len(hit_list)}, rep_factor: {r}')
		print(f'rank:{partition} miss_list_len: {len(miss_list)}')

		# if rep_factor is a multiple of 0.5
		if r % 1 != 0:
			r = r - 0.5
			r = int(r)
			hit_samps = len(hit_list) * r + len(hit_list)//2
			miss_samps = needtotake - hit_samps

			# print(f'hit_samps: {hit_samps}')
			# print(f'miss_samps: {miss_samps}')

			art_hit_list = hit_list*r + hit_list[:len(hit_list)//2]
			art_miss_list = miss_list

			random.shuffle(art_hit_list)
			random.shuffle(art_miss_list)
		else:
			r = int(r)
			hit_samps = len(hit_list) * r
			miss_samps = needtotake - hit_samps

			# print(f'hit_samps: {hit_samps}')
			# print(f'miss_samps: {miss_samps}')

			art_hit_list = hit_list*r 
			art_miss_list = miss_list

			random.shuffle(art_hit_list)
			random.shuffle(art_miss_list)

		return art_hit_list,art_miss_list,miss_samps


	def trace_partition(self, data_map_file):
		"""Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
		logging.info(f"Partitioning data by profile {data_map_file}...")

		clientId_maps = {} # key = sample_id, value = unique id of client
		unique_clientIds = {} # client_ids are numbered as 0,1,2,3. for ex - 9800,237,124 would be identified as 0,1,2,
		# load meta data from the data_map_file
		with open(data_map_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			read_first = True
			sample_id = 0

			for row in csv_reader:
				if read_first:
					logging.info(f'Trace names are {", ".join(row)}')
					read_first = False
				else:
					client_id = row[0]

					if client_id not in unique_clientIds:
						unique_clientIds[client_id] = len(unique_clientIds)

					clientId_maps[sample_id] = unique_clientIds[client_id]
					self.client_label_cnt[unique_clientIds[client_id]].add(
						row[-1]) # which labels are owned by a particular client.
					sample_id += 1


		clientpath = os.path.join('/home/cc', 'client')
		clientmappath = os.path.join(clientpath,"clientmap.csv")

		#logging.info(f"unique_clientIds before function call {unique_clientIds}")
		self.save_client_mapping(unique_clientIds, clientmappath)
		# Partition data given mapping
		self.partitions = [[] for _ in range(len(unique_clientIds))]

		for idx in range(sample_id):
			self.partitions[clientId_maps[idx]].append(idx)

	def partition_data_helper(self, num_clients, data_map_file=None):

		# read mapping file to partition trace
		if data_map_file is not None:
			self.trace_partition(data_map_file)
		else:
			self.uniform_partition(num_clients=num_clients)

	def uniform_partition(self, num_clients):
		# random partition
		numOfLabels = self.getNumOfLabels()
		data_len = self.getDataLen()
		logging.info(f"Randomly partitioning data, {data_len} samples...")

		indexes = list(range(data_len))
		self.rng.shuffle(indexes)

		for _ in range(num_clients):
			part_len = int(1./num_clients * data_len)
			self.partitions.append(indexes[0:part_len])
			indexes = indexes[part_len:]

	def use(self, partition, istest):

		clientpath = os.path.join('/home/cc', 'client')

		weightpath = os.path.join(clientpath, str(partition+1) + '.csv')


		client_numcall_path = os.path.join(clientpath,str(partition+1) + 'numcall.csv')

		if not istest:
			print(f"rank {partition + 1} getting called. {weightpath} for Train")
			resultIndex = self.partitions[partition] #this is an array of sample indexes for the client with a paritcular rank

			copy_orig_indices = resultIndex

			# clientpath = os.path.join(parser.args.log_path, "logs", parser.args.job_name,
			#               parser.args.time_stamp, 'client')
			
			#print(f"rank {partition + 1} getting called. {weightpath} for Train")


				#print(f"curr_map rank {partition + 1} exists, keys: {len(curr_map.keys())} , {curr_map}")



			exeuteLength = len(resultIndex) if not istest else int(
				len(resultIndex) * self.args.test_ratio) # use all the samples of the client or according to a particular ratio
			resultIndex = resultIndex[:exeuteLength] #sample indices of the client.

			#logging.info(f"rank {partition + 1} has {len(resultIndex)} indices. {resultIndex}")

			if os.path.exists(weightpath) and os.path.isfile(weightpath):
				curr_map = self.read_imp_freq_mapping_from_csv(weightpath)
			else:
				samplelist = resultIndex
				weightlist = [1.0]* len(resultIndex)
				freqlist = [0.0]* len(resultIndex)
				if not os.path.exists(clientpath):
					os.makedirs(clientpath) 
				#self.save_imp_mapping_to_csv(samplelist, weightlist, weightpath)
				self.save_imp_freq_mapping_to_csv(samplelist, weightlist, freqlist, weightpath)
				curr_map = self.read_imp_freq_mapping_from_csv(weightpath)

			tensor_samples = torch.tensor(list(curr_map.keys()))
			#tensor_weights = torch.tensor(list(curr_map.values()), dtype = torch.float)
			curr_map_vals = curr_map.values()
			curr_map_weights = [val[0] for val in curr_map_vals]
			
			############ sample experience ####################
			curr_map_freq = [val[1] for val in curr_map_vals]
			max_w, min_w = max(curr_map_weights), min(curr_map_weights)
			max_f, min_f = max(curr_map_freq), min(curr_map_freq)

			# print(f"rank: {partition + 1}, curr_map_weights: {curr_map_weights}, curr_map_freq: {curr_map_freq}")
			# print(f"rank: {partition + 1}, max_w: {max_w}, type: {type(max_w)}, max_f: {max_f}, type: {type(max_f)}")

			# we want highest loss and lowest freq. highest loss means this sample can contribute more to building global acc, 
			# and lowest freq means this important sample needs more training to become experienced
			# if samples become more experienced, client becomes more experienced, we have a more experienced team.
			try:
				normw = [(item - min_w) / (max_w - min_w) for item in curr_map_weights] #
				#normf = [(max_f - item) / (max_f - min_f) for item in curr_map_freq] #
				normf = [(item - min_f) / (max_f - min_f) for item in curr_map_freq] #
				sample_experience = [curr_map_weights[i] + normw[i] + 0*normf[i] for i in range(len(normw))]
			except ZeroDivisionError:
				sample_experience = [val[0] for val in curr_map_vals]

			if self.args.subs == 1.0 or self.args.fedcaseimp == 1.0:
				sample_experience = [val[0] for val in curr_map_vals]

			curr_map_weights = sample_experience

			#print(f"rank: {partition + 1} weights: {curr_map_weights}")
			###################################################

			tensor_weights = torch.tensor(curr_map_weights, dtype = torch.float)

			ls_param_path = os.path.join(clientpath, str(partition+1) + '_param.csv')

			if os.path.exists(ls_param_path) and os.path.isfile(ls_param_path):
				loss, ls_param = self.read_param(ls_param_path)
			else:
				ls_param = 1e-2

			# if ls_param > 1:
			#     loss = 1e-4
			#     ls_param = 1e-2

			if os.path.exists(client_numcall_path) and os.path.isfile(client_numcall_path):
				clientnumcalls = self.read_client_numcalls(client_numcall_path)
			else:
				clientnumcalls = 0

			clientnumcalls +=1

			#print(f"rank {partition+1} tensor weights: {tensor_weights}")
			#print(f"rank {partition + 1} getting called. {tensor_weights} for Train")
			#if self.args != 1:
			try:
				imp_idxes = torch.multinomial(tensor_weights.add(ls_param), len(resultIndex), replacement = True)
				actual_indices = tensor_samples[imp_idxes]
				actual_indices = actual_indices.tolist()
				#print(f"rank {partition+1} tensor weights: {actual_indices}, type(actual_indices[0]): {type(actual_indices[0])}")
				resultIndex = actual_indices
			except:
				print(f"multinomial error raised.")
				actual_indices = resultIndex

			norm_loss_indices = resultIndex

			if clientnumcalls % 3 == 0 and clientnumcalls !=0:
				#pads_sort gets called
				print(f"rank {partition + 1} undergoing pads_sort. clientnumcalls: {clientnumcalls}")
				resultIndex = self.pads_sort(resultIndex)

			needtotake = self.args.batch_size * self.args.local_steps

			if needtotake > len(resultIndex): 
				needtotake = len(resultIndex)

			if self.args.fedcase == 0.0:
				resultIndex = resultIndex[:needtotake]

			#if self.args.subs == 1.0:
			cache_hit_list, cache_miss_list, num_miss_samps = self.prepare_hits( str(int(partition)+1),resultIndex, needtotake, self.args.rep_factor)

			logging.info(f"rank {partition + 1} has {len(cache_hit_list)} hits, and {len(cache_miss_list)} misses.")

			### Need sampling logic here
			# logging.info(f"rank {partition} is getting called.")
			# if partition in self.checker:
			#     self.checker[partition] +=1
			#     logging.info(f"clientId(rank): {partition} : self.checker : {self.checker[partition]}")
			# else:
			#     self.checker[partition] = 1
			#     logging.info(f"rank1stinitialize: {partition}")


			####
			resultIndex = cache_hit_list + cache_miss_list[:num_miss_samps]

			resultIndex = resultIndex[:needtotake]

			norm_loss_indices = norm_loss_indices[:needtotake]

			# clientnumcalls +=1

			self.write_client_numcalls(str(partition+1), clientnumcalls, client_numcall_path)
			#self.rng.shuffle(resultIndex)
			#print(f"rank {partition} getting called. {resultIndex}")
			# print(f"baseline value: {self.args.base_case}, cache_traindata: {self.args.cache_traindata}, cache_testdata: {self.args.cache_testdata}, use_cuda: {self.args.use_cuda}, host_ip: {self.args.host_ip}")
			# if self.args.base_case:
			#     print(f"inside: {self.args.base_case}, cache_traindata: {self.args.cache_traindata}, cache_testdata: {self.args.cache_testdata}, use_cuda: {self.args.use_cuda}, host_ip: {self.args.host_ip}")
			#     if self.args.base_case == False:
			#         print(f"baseline val: {self.args.base_case}")

			if self.args.base_case:
				exeuteLength = len(copy_orig_indices) if not istest else int(
					len(copy_orig_indices) * self.args.test_ratio) # use all the samples of the client or according to a particular ratio
				resultIndex = copy_orig_indices[:exeuteLength] #sample indices of the client.
				self.rng.shuffle(resultIndex)
				if self.args.subs == 1.0:
					#resultIndex = resultIndex[:needtotake]
					cache_hit_list, cache_miss_list, num_miss_samps = self.prepare_hits( str(int(partition)+1),resultIndex, needtotake, self.args.rep_factor)
					resultIndex = cache_hit_list + cache_miss_list[:num_miss_samps]
					logging.info(f"inside base: rank {partition + 1} has {len(cache_hit_list)} hits, and {len(cache_miss_list)} misses.")
					#logging.info(f"inside base: rank {partition + 1}: hitlist: {cache_hit_list}")
				resultIndex = resultIndex[:needtotake]
				# if len(resultIndex) == len(set(resultIndex)):
				#     print(f"baseline getting executed, client {partition + 1}: resultIndex all unique elements.")
			if len(resultIndex) == len(set(resultIndex)):
				print(f"client {partition + 1}: resultIndex all unique elements.")

			self.rng.shuffle(copy_orig_indices)
			copy_orig_indices = copy_orig_indices[:needtotake]
			previous_set = set(copy_orig_indices)
			new_set = set(resultIndex)
			common_elements = len(previous_set.intersection(new_set))
			uncommon_elements = needtotake - common_elements
			print(f"client {partition+1} difference between data sampling methods: {uncommon_elements}")

			if self.args.mer_indices == 1.0:
				resultIndex = norm_loss_indices

			return Partition(self.data, resultIndex) #self.data is all of the actual data samples i.e. jpg or png images from femnist.py. resultIndex is the image indices for the particular client.
		else:
			print(f"rank {partition + 1} getting called. {weightpath} for Test")

			resultIndex = self.partitions[partition] #this is an array of sample indexes for the client with a paritcular rank
			exeuteLength = len(resultIndex) if not istest else int(
				len(resultIndex) * self.args.test_ratio) # use all the samples of the client or according to a particular ratio
			resultIndex = resultIndex[:exeuteLength] #sample indices of the client.
			self.rng.shuffle(resultIndex)
			#print(f"rank {partition} getting called. {resultIndex}")

			return Partition(self.data, resultIndex)


	def getSize(self):
		# return the size of samples
		return {'size': [len(partition) for partition in self.partitions]}


def select_dataset(rank, partition, batch_size, args, isTest=False, collate_fn=None, indices=None, weights=None):
	"""Load data given client Id"""
	partition = partition.use(rank - 1, isTest) #gives me the actual dataset with images instead of a list of locations of images or samples index number.
	dropLast = False if isTest else True
	if isTest:
		num_loaders = 0
	else:
		num_loaders = min(int(len(partition)/args.batch_size/2), args.num_loaders)
	if num_loaders == 0:
		time_out = 0
	else:
		time_out = 60

	#print(f"rank {rank} partition: {partition}, type(partition[0]): {type(partition[0])}")
	if collate_fn is not None:
		return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
	return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)
