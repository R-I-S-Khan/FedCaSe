from __future__ import print_function

import csv
import os
import os.path
import warnings

import time
from datetime import datetime
import argparse

import random
import PIL.Image as Image
import numpy as np
import redis
import io
from io import BytesIO
import numpy as np

import redis
import heapdict
import PIL
from rediscluster import RedisCluster
from collections import OrderedDict
from heapdict import heapdict
import ast

class FEMNIST():
	"""
	Args:
		root (string): Root directory of dataset where ``MNIST/processed/training.pt``
			and  ``MNIST/processed/test.pt`` exist.
		train (bool, optional): If True, creates dataset from ``training.pt``,
			otherwise from ``test.pt``.
		download (bool, optional): If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
	"""

	classes = []

	@property
	def train_labels(self):
		warnings.warn("train_labels has been renamed targets")
		return self.targets

	@property
	def test_labels(self):
		warnings.warn("test_labels has been renamed targets")
		return self.targets

	@property
	def train_data(self):
		warnings.warn("train_data has been renamed data")
		return self.data

	@property
	def test_data(self):
		warnings.warn("test_data has been renamed data")
		return self.data

	def __init__(self, root, dataset='train', transform=None, target_transform=None, imgview=False, cache_data = False,
		PQ=None, ghost_cache=None, key_counter= None, 
		wss = 1.0, host_ip = '0.0.0.0', port_num = '6379'):

		self.data_file = dataset  # 'train', 'test', 'validation'
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.path = os.path.join(self.processed_folder, self.data_file) #this path formation is of no use.

		# load data and targets
		#self.data, self.targets = self.load_file(self.path) #self.path is not used. check self.load_file.
		#load clients, corresponding data and targets
		self.clients, self.data, self.targets = self.load_file(self.path) #self.path is not used. check self.load_file.
		#self.data, self.targets = self.load_file(self.path) #self.path is not used. check self.load_file.
		#self.mapping = {idx:file for idx, file in enumerate(raw_data)}

		self.imgview = imgview
		self.wss = wss
		self.cache_data = cache_data
		#temporary addition for evaluation of other caching policies
		self.barrier = 0
		self.random_evict = 0
		self.policy = 'fedcaseimp' #'shade' #'baselfu' #'lfu' #'lru' #'shade' #fedcase #lfu #lru

		# self.cache_portion = self.wss * len(self.data)
		# self.cache_portion = int(self.cache_portion // 1)

		self.clientmappath = os.path.join("/home/cc/client", "clientmap.csv")

		if host_ip == '0.0.0.0':
			self.key_id_map = redis.Redis()
		else:
			self.startup_nodes = [{"host": host_ip, "port": port_num}]
			self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
			#this method is accessed whenever a client has a to fetch a data sample for training.
		"""

		imgName, target = self.data[index], int(self.targets[index])

		if self.data_file == 'train' or self.data_file == 'validation':
			fetched_from_cache = 0

			#print(f'type of client before: {type(client)}')
			client = self.clients[index]
			#print(f'client {self.clients[index]} is accessing image {index}: {imgName}') 
			client = int(client)
			#print(f'type of index: {type(index)}')
			#print(f'type of client after: {type(client)}')

			client_mapping = self.read_client_mapping(self.clientmappath)

			client_mapped_id = str(int(client_mapping[client]) + 1)

			clientpath = os.path.join('/home/cc', 'client')

			weightpath = os.path.join(clientpath, str(client_mapped_id) + '.csv')

			ghostcachepath = os.path.join(clientpath, str(client_mapped_id) + '_ghost.csv')

			cur_cache_path = os.path.join(clientpath, str(client_mapped_id) + '_curr.csv')

			#all_client_samples = self.read_imp_mapping_from_csv(weightpath)
			all_client_samples = self.read_imp_freq_mapping_from_csv(weightpath)


			curr_ghost_map = {}

			if os.path.exists(cur_cache_path) and os.path.isfile(cur_cache_path):
				client_cur_cache_samples = self.read_list_from_csv(cur_cache_path)
			else:
				client_cur_cache_samples = []  

			if os.path.exists(ghostcachepath) and os.path.isfile(ghostcachepath):   
				#curr_ghost_map = self.read_imp_mapping_from_csv(ghostcachepath)
				curr_ghost_map = self.read_imp_freq_mapping_from_csv(ghostcachepath)
				# client_cache_key_count = len(curr_ghost_map)
				# client_cache_key_ids = list(curr_ghost_map.keys())

			self.cache_portion = self.wss * len(all_client_samples)
			self.cache_portion = int(self.cache_portion // 1)

			#print(f"client_mapped_id: {client_mapped_id}")

			#print(f"curr_ghost_map formed: {curr_ghost_map}")

			redis_client_cache_key_count, redis_client_cache_key_ids = self.client_keys_in_cache(client_mapped_id + "_")
			#print(f"client: {client_mapped_id}, redis_client_cache_key_count: {redis_client_cache_key_count}, redis_client_cache_key_ids: {redis_client_cache_key_ids}")
			redis_client_cache_key_vals = []
			mintstamp = 999999999999
			#client_cache_key_count, client_cache_key_ids = len(client_cur_cache_samples), client_cur_cache_samples
			try:
				for keyid in redis_client_cache_key_ids:
					#print(f"client: {client_mapped_id}, keyid: {keyid}")
					return_tup = self.key_id_map.hgetall(keyid)
					tstamp = int(return_tup[b'timestamp'].decode('utf-8'))
					#tstamp = int(self.key_id_map.hgetall(keyid)[b'timestamp'].decode('utf-8'))
					#print(f"client: {client_mapped_id}, tstamp: {tstamp}")
					if tstamp < mintstamp:
						mintstamp = tstamp
						minkey = keyid
						#print(f"client: {client_mapped_id}, changed. mintstamp: {mintstamp}, minkey: {minkey}")
					#mintstamp = min(mintstamp, tstamp)
					#redis_client_cache_key_vals.append(int(self.key_id_map.get(keyid)[b'timestamp'].decode('utf-8')))
					redis_client_cache_key_vals.append(tstamp)
					#print(f"client: {client_mapped_id}, redis_client_cache_key_vals: {redis_client_cache_key_vals}")
				#redis_client_cache_key_vals = [int(self.key_id_map.get(keyid)[b'timestamp'].decode('utf-8')) for keyid in redis_client_cache_key_ids]
				currtimestamp = max(redis_client_cache_key_vals) + 1
				leastrecentused_timestamp = mintstamp
				leastrecentused_key = minkey
				#print(f"client: {client_mapped_id}, redis_client_cache_key_vals: {redis_client_cache_key_vals}")
				#print(f"client: {client_mapped_id}, leastrecentused_key: {leastrecentused_key}, leastrecentused_timestamp: {leastrecentused_timestamp}")
			except:
				currtimestamp = 1

			#random_id = 

			redis_client_cache_key_ids = [keyid.replace(client_mapped_id + '_', '') for keyid in redis_client_cache_key_ids]
			client_cache_key_count, client_cache_key_ids = redis_client_cache_key_count, redis_client_cache_key_ids

			if client_cache_key_count > 0: 
				random_id = str(random.choice(client_cache_key_ids))
				random_key = client_mapped_id + '_'+ random_id

			#redis_client_cache_key_ids.sort()
			#client_cache_key_ids.sort()

			# if redis_client_cache_key_ids == client_cache_key_ids:
			#     print(f"client_mapped_id: {client_mapped_id}, redis = csv")
			# else:
			#     print(f"client_mapped_id: {client_mapped_id}, redis and csv not equal")

			# print(f"client_mapped_id: {client_mapped_id}, redis_client_cache_key_count: {redis_client_cache_key_count}, client_cache_key_count: {client_cache_key_count}")
			# print(f"client_mapped_id: {client_mapped_id}, redis_client_cache_key_ids: {redis_client_cache_key_ids}, client_cache_key_ids: {client_cache_key_ids}")            

			#print(f"client_keys_in_cache worked: {client_cache_key_count} , {client_cache_key_ids}")
			#print(f"client_keys_in_cache worked: ID: {client_mapped_id}, cache_portion: {self.cache_portion}, keys: {client_cache_key_count}")

			#client_cache_key_ids = [keyid.replace(client_mapped_id + '_', '') for keyid in client_cache_key_ids]

			#print(f"new client_cache_key_ids without prefix {client_cache_key_ids}")
			if os.path.exists(ghostcachepath) and os.path.isfile(ghostcachepath) and len(curr_ghost_map) > 0: 

				#print(f"curr_ghost_map: {curr_ghost_map}")
				client_curr_cache_dict = self.form_mapping_of_client_keys(client_cache_key_ids, all_client_samples)
				client_ghost_cache_heap = self.form_heapdict(curr_ghost_map)
			else:
				client_curr_cache_dict = self.form_mapping_of_client_keys(client_cache_key_ids, all_client_samples)
				client_curr_cache_heap = self.form_heapdict(client_curr_cache_dict)
				client_ghost_cache_heap = client_curr_cache_heap

			#print(f"client_curr_cache_dict formed: {client_curr_cache_dict}")

			client_curr_cache_heap = self.form_heapdict(client_curr_cache_dict)

			#for after warm-up cases.
			if len(curr_ghost_map) < len(client_curr_cache_dict) :
				#print(f"client_ghost_cache_heap is the same as client_curr_cache_heap now")
				client_ghost_cache_heap = client_curr_cache_heap

			ghost_map_dict = dict(client_ghost_cache_heap)
			ghost_map_dict_weights = [float(val[0]) for val in ghost_map_dict.values()]
			ghost_map_dict_freqs = [float(val[1]) for val in ghost_map_dict.values()]
			if len(ghost_map_dict_weights) > 0 and len(ghost_map_dict_freqs) > 0:
				maxw,minw = max(ghost_map_dict_weights), min(ghost_map_dict_weights)
				maxf,minf = max(ghost_map_dict_freqs), min(ghost_map_dict_freqs)

			# print(f"client_mapped_id: {client_mapped_id}, ghost_map_dict_weights: {ghost_map_dict_weights}, ghost_map_dict_freqs: {ghost_map_dict_freqs}")
			# print(f"client_mapped_id: {client_mapped_id}, maxw: {maxw},type: {type(maxw)} minw:{minw}, type:{type(minw)}, maxf:{maxf}, type: {type(maxf)} minf: {minf}, type: {type(minf)}")

			#print(f"client_curr_cache_heap formed: {client_curr_cache_heap}")

			# print(f"client_mapped_id: {client_mapped_id}, keys_cnt: {keys_cnt}, cache_portion: {self.cache_portion}")

			# for key, value in client_curr_cache_heap.items():
			#     print(f"client_mapped_id: {client_mapped_id}, client_curr_cache_heap: K:{key}, typekey: {type(key)}, V:{value}, type_val[0]: {type(value[0])}, type_val[1]: {type(value[1])}")

			# peek_item = client_curr_cache_heap.peekitem()
			# print(f"client_mapped_id: {client_mapped_id}, peek_item: {peek_item}")

			# for key, value in client_ghost_cache_heap.items():
			#     print(f"client_mapped_id: {client_mapped_id}, client_ghost_cache_heap: K:{key}, typekey: {type(key)}, V:{value}, type_val[0]: {type(value[0])}, type_val[1]: {type(value[1])}")


			# for key, value in client_curr_cache_heap.items():
			#     print(f" client_curr_cache_heap: K:{key}, V:{value}")

			# if curr_ghost_map[index] > client_curr_cache_heap.peekitem()


			#prepare a heapdict with the client_cache_key_ids.

			#check the curr sample's imp against the 

			######## CACHING LOGIC GOES HERE #######

			if self.cache_data and self.key_id_map.exists(client_mapped_id + "_" + str(index)):
				try:
					#print('hitting %d' %(index))
					#print(f"{client_mapped_id} hitting {index}")
					#byte_image = self.key_id_map.get(client_mapped_id + "_" + str(index))
					return_tup = self.key_id_map.hgetall(client_mapped_id + "_" + str(index))
					byte_image = return_tup[b'imgbyte']
					image_timestamp = int(return_tup[b'timestamp'].decode('utf-8'))
					updated_timestamp = currtimestamp
					tup_val = {"imgbyte": byte_image, "timestamp": currtimestamp}
					self.key_id_map.hmset(client_mapped_id + "_" + str(index), tup_val)

					byteImgIO = io.BytesIO(byte_image)
					img = Image.open(byteImgIO)
					img = img.convert('RGB')
					fetched_from_cache = 1

					updated_tup = self.key_id_map.hgetall(client_mapped_id + "_" + str(index))
					updated_timestamp = int(updated_tup[b'timestamp'].decode('utf-8'))
					ret_byte_image = updated_tup[b'imgbyte']
					# if byte_image == ret_byte_image:
					# 	print(f"{client_mapped_id} : {index} , byte_image and ret_byte_image are the same.")
					# print(f"{client_mapped_id} hitting {index}, prevtimestamp: {image_timestamp}, currtimestamp: {currtimestamp}, updated_timestamp: {updated_timestamp}")
				except: #PIL.UnidentifiedImageError:
					try:
						print("Could not open image in path from byteIO: ", path)
						img= Image.open(os.path.join(self.root, imgName))
						img = img.convert('RGB')
						print("Successfully opened file from path using open.")
					except:
						print("Could not open even from path. The image file is corrupted.")
			else:

			# doing this so that it is consistent with all other datasets
			# to return a PIL Image
				#print(f"{client_mapped_id} missing {index}")
				img = Image.open(os.path.join(self.root, imgName))

				#keys_cnt = self.key_counter + 50
				keys_cnt = client_cache_key_count
				#self.cache_portion = 5
				#print(f"client_mapped_id: {client_mapped_id}, keys_cnt: {keys_cnt}, cache_portion: {self.cache_portion}")
				#temporary addition for evaluation of other caching policies
				if keys_cnt >= self.cache_portion and self.random_evict == 1:
					if self.key_id_map.exists(random_key):
						self.key_id_map.delete(random_key)
						keys_cnt-=1
						client_cur_cache_samples.remove(str(random_id))
						self.write_list_to_csv(client_cur_cache_samples, cur_cache_path)

				# while keys_cnt >= self.cache_portion:
				#     rnd = random.choice(client_cache_key_ids)
				#     random_id = str(rnd)
				#     random_key = client_mapped_id + '_'+ random_id
				#     if self.key_id_map.exists(random_key):
				#         self.key_id_map.delete(random_key)
				#         keys_cnt-=1
				#         client_cache_key_ids.remove(rnd)
				#         try:
				#             client_cur_cache_samples.remove(str(random_id))
				#             self.write_list_to_csv(client_cur_cache_samples, cur_cache_path)
				#         except:
				#             continue

				if(keys_cnt >= self.cache_portion and self.barrier == 0.0 and self.policy == 'lru'):
					#print(f"inside lru eviction")
					#print(f"leastrecentused_key: {leastrecentused_key}")
					#print(f"does it exist? {self.key_id_map.exists(leastrecentused_key)}")

					try:
						if self.key_id_map.exists(leastrecentused_key):
							self.key_id_map.delete(leastrecentused_key)
							print(f"client: {client_mapped_id}, lru working. {leastrecentused_key}:{leastrecentused_timestamp} deleted. {str(index)}:{currtimestamp} inserted.")
							keys_cnt-=1
							underscore_index = leastrecentused_key.index('_')
							leastrecentused_keycurr = leastrecentused_key[underscore_index + 1:]
							#print(f"client: {client_mapped_id}, before client_cur_cache_samples: {client_cur_cache_samples}")
							client_cur_cache_samples.remove(str(leastrecentused_keycurr))
							#print(f"client: {client_mapped_id}, after client_cur_cache_samples: {client_cur_cache_samples}")
							self.write_list_to_csv(client_cur_cache_samples, cur_cache_path)
					except:
						#print("Could not evict item or PQ was empty or PQ did not have that item.")
						pass
				if(keys_cnt >= self.cache_portion and self.barrier == 0.0 and self.policy == 'baselfu'):
					try:
						#print("Check if we need to evict:")
						peek_item = client_curr_cache_heap.peekitem()
						#print(f"client_mapped_id: {client_mapped_id}, peek_item: {peek_item}, curr_item: {index}")

						peek_item_weight = peek_item[1][0]
						peek_item_freq = peek_item[1][1]
						
						curr_item_weight = client_ghost_cache_heap[str(index)][0]
						curr_item_freq = client_ghost_cache_heap[str(index)][1]

						#print(f"client_mapped_id: {client_mapped_id}, peek_item_weight: {peek_item_weight}, type: {type(peek_item_weight)} peek_item_freq: {peek_item_freq}, type: {type(peek_item_freq)} curr_item_weight: {curr_item_weight}, type: {type(curr_item_weight)} , curr_item_freq:{curr_item_freq}, type: {type(curr_item_freq)}")
						peek_item_weight = float(peek_item_weight)
						peek_item_freq = float(peek_item_freq)
						curr_item_weight = float(curr_item_weight)
						curr_item_freq = float(curr_item_freq)
						#print(f"just before calc: client_mapped_id: {client_mapped_id}, maxw: {maxw},type: {type(maxw)} minw:{minw}, type:{type(minw)}, maxf:{maxf}, type: {type(maxf)} minf: {minf}, type: {type(minf)}")
						if (maxw - minw) > 0 and (maxf - minf) > 0:
							peek_item_experience_weight = (peek_item_weight - minw) / (maxw - minw)
							#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience_weight: {peek_item_experience_weight}, type: {type(peek_item_experience_weight)}")
							#peek_item_experience_freq =  (maxf - peek_item_freq ) / (maxf - minf)
							peek_item_experience_freq =  (peek_item_freq - minf) / (maxf - minf)
							#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience_freq: {peek_item_experience_freq}, type: {type(peek_item_experience_freq)}")
							curr_item_experience_weight = (curr_item_weight - minw) / (maxw - minw)
							#print(f"client_mapped_id: {client_mapped_id}, curr_item_experience_weight: {curr_item_experience_weight}, type: {type(curr_item_experience_weight)}")
							#curr_item_experience_freq =  (maxf - curr_item_freq ) / (maxf - minf)
							curr_item_experience_freq =  (curr_item_freq - minf) / (maxf - minf)
							#print(f"client_mapped_id: {client_mapped_id}, curr_item_experience_freq: {curr_item_experience_freq}, type: {type(curr_item_experience_freq)}")
							peek_item_experience = peek_item_experience_weight + peek_item_experience_freq
							curr_item_experience = curr_item_experience_weight + curr_item_experience_freq
						else:
							peek_item_experience = peek_item
							curr_item_experience = client_ghost_cache_heap[str(index)]
						
						evicted_item = client_curr_cache_heap.popitem()
						
						# print(f"client_mapped_id: {client_mapped_id}, Inserting Index: {index}, value: {client_ghost_cache_heap[str(index)]}, curr_experience: {curr_item_experience}")
						# print(f"client_mapped_id: {client_mapped_id}, Evicting index: {evicted_item[0]} Weight: {evicted_item[1][0]} Frequency: {evicted_item[1][1]}, evict_experience: {peek_item_experience}")

						#print(f"before removing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")
						client_cur_cache_samples.remove(str(evicted_item[0]))
						#print(f"after removing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")

						self.write_list_to_csv(client_cur_cache_samples, cur_cache_path)

						key_id_map_ind = client_mapped_id + '_'+ str(evicted_item[0])

						if self.key_id_map.exists(key_id_map_ind):
							#print(f"deleting {key_id_map_ind} from client_mapped_id: {client_mapped_id}")
							self.key_id_map.delete(key_id_map_ind)
							#print(f"deleting {key_id_map_ind} from client_mapped_id: {client_mapped_id}")

						keys_cnt-=1

					except:
						#print("Could not evict item or PQ was empty or PQ did not have that item.")
						pass

				if(keys_cnt >= self.cache_portion and self.barrier == 0.0 and self.policy == 'lfu'):
					try:
						#print("Check if we need to evict:")
						peek_item = client_curr_cache_heap.peekitem()
						#print(f"client_mapped_id: {client_mapped_id}, peek_item: {peek_item}, curr_item: {index}")

						peek_item_weight = peek_item[1][0]
						peek_item_freq = peek_item[1][1]
						# print(f"does item exist in ghost map yet?")
						# if str(index) in client_ghost_cache_heap:
						#     print(f"client_mapped_id: {client_mapped_id}, yes {str(index)} exists in client_ghost_cache_heap")
						# else:
						#     print(f"client_mapped_id: {client_mapped_id}, no {str(index)} does not exist in client_ghost_cache_heap")
						# print(f"client_mapped_id: {client_mapped_id}, curr_item_weights: {client_ghost_cache_heap[str(index)]}")
						curr_item_weight = client_ghost_cache_heap[str(index)][0]
						curr_item_freq = client_ghost_cache_heap[str(index)][1]

						#print(f"client_mapped_id: {client_mapped_id}, peek_item_weight: {peek_item_weight}, type: {type(peek_item_weight)} peek_item_freq: {peek_item_freq}, type: {type(peek_item_freq)} curr_item_weight: {curr_item_weight}, type: {type(curr_item_weight)} , curr_item_freq:{curr_item_freq}, type: {type(curr_item_freq)}")
						peek_item_weight = float(peek_item_weight)
						peek_item_freq = float(peek_item_freq)
						curr_item_weight = float(curr_item_weight)
						curr_item_freq = float(curr_item_freq)
						#print(f"just before calc: client_mapped_id: {client_mapped_id}, maxw: {maxw},type: {type(maxw)} minw:{minw}, type:{type(minw)}, maxf:{maxf}, type: {type(maxf)} minf: {minf}, type: {type(minf)}")
						if (maxw - minw) > 0 and (maxf - minf) > 0:
							peek_item_experience_weight = (peek_item_weight - minw) / (maxw - minw)
							#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience_weight: {peek_item_experience_weight}, type: {type(peek_item_experience_weight)}")
							#peek_item_experience_freq =  (maxf - peek_item_freq ) / (maxf - minf)
							peek_item_experience_freq =  (peek_item_freq - minf) / (maxf - minf)
							#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience_freq: {peek_item_experience_freq}, type: {type(peek_item_experience_freq)}")
							curr_item_experience_weight = (curr_item_weight - minw) / (maxw - minw)
							#print(f"client_mapped_id: {client_mapped_id}, curr_item_experience_weight: {curr_item_experience_weight}, type: {type(curr_item_experience_weight)}")
							#curr_item_experience_freq =  (maxf - curr_item_freq ) / (maxf - minf)
							curr_item_experience_freq =  (curr_item_freq - minf) / (maxf - minf)
							#print(f"client_mapped_id: {client_mapped_id}, curr_item_experience_freq: {curr_item_experience_freq}, type: {type(curr_item_experience_freq)}")
							peek_item_experience = peek_item_experience_weight + peek_item_experience_freq
							curr_item_experience = curr_item_experience_weight + curr_item_experience_freq
						else:
							peek_item_experience = peek_item
							curr_item_experience = client_ghost_cache_heap[str(index)]

						#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience: {peek_item_experience}, curr_item_experience: {curr_item_experience}, type: {type(peek_item_experience)}, type: {type(curr_item_experience)}")
						#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience: {peek_item_experience}, curr_item_experience: {curr_item_experience}")
						# peek_item_experience = peek_item_experience_weight + peek_item_experience_freq
						# curr_item_experience = curr_item_experience_weight + curr_item_experience_freq

						#print(f"client_ghost_cache_heap {client_ghost_cache_heap}")
						# for key, value in client_ghost_cache_heap.items():
						#     print(f" client_ghost_cache_heap: K:{key}, V:{value}")
						#print(f"client_ghost_cache_heap[{index}] : {client_ghost_cache_heap[index]} , {peek_item}")
						#print(f"peek_item {peek_item}, peek_item[1]: {peek_item[1]}, type: {type(peek_item[1])}")
						#print(f"index {index}, type: {type(index)}")
						#print(f"index: {index}, imp val: {client_ghost_cache_heap[str(index)]}, type: {type(client_ghost_cache_heap[str(index)])}")
						#print(f"curr len of client_curr_cache_heap: {len(client_curr_cache_heap)}")
						#if client_ghost_cache_heap[str(index)] > peek_item[1]: 
						#if curr_item_experience > peek_item_experience:
						if curr_item_freq > peek_item_freq:
							evicted_item = client_curr_cache_heap.popitem()
							#print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))
							#print(f"evicted item {evicted_item}")
							#print(f"Evicting index: {evicted_item[0]} Weight: {evicted_item[1]}")# %(evicted_item[0], evicted_item[1]))
							print(f"client_mapped_id: {client_mapped_id}, Inserting Index: {index}, value: {client_ghost_cache_heap[str(index)]}, curr_experience: {curr_item_experience}")
							print(f"client_mapped_id: {client_mapped_id}, Evicting index: {evicted_item[0]} Weight: {evicted_item[1][0]} Frequency: {evicted_item[1][1]}, evict_experience: {peek_item_experience}")

							#print(f"before removing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")
							client_cur_cache_samples.remove(str(evicted_item[0]))
							#print(f"after removing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")

							self.write_list_to_csv(client_cur_cache_samples, cur_cache_path)

							# client_cur_cache_samples = self.read_list_from_csv(cur_cache_path)
							# print(f"after removing and writing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")

							key_id_map_ind = client_mapped_id + '_'+ str(evicted_item[0])

							# prefix = client_mapped_id + '_'
							# all_keys = self.key_id_map.keys()
							# all_keys_string = [key.decode('ascii') for key in all_keys]
							# matching_keys = [key for key in all_keys_string if key.startswith(prefix)]
							# print(f"before removing from redis. clientID: {client_mapped_id}, len(matching_keys): {len(matching_keys)}, keys: {matching_keys}")
							if self.key_id_map.exists(key_id_map_ind):
								#print(f"deleting {key_id_map_ind} from client_mapped_id: {client_mapped_id}")
								self.key_id_map.delete(key_id_map_ind)
								# if self.key_id_map.exists(key_id_map_ind):
								#     print(f"{key_id_map_ind} from client_mapped_id: {client_mapped_id} not deleted.")
								# else:
								#     print(f"{key_id_map_ind} from client_mapped_id: {client_mapped_id} deleted Successfully")

							#print(f"deleted {key_id_map_ind}")
							# all_keys = self.key_id_map.keys()
							# all_keys_string = [key.decode('ascii') for key in all_keys]
							# matching_keys = [key for key in all_keys_string if key.startswith(prefix)]
							# print(f"after removing from redis. clientID: {client_mapped_id}, len(matching_keys): {len(matching_keys)}, keys: {matching_keys}")
							keys_cnt-=1

					except:
						#print("Could not evict item or PQ was empty or PQ did not have that item.")
						pass

				if(keys_cnt >= self.cache_portion and self.barrier == 0.0 and self.policy == 'shade'):
					try:
						#print("Check if we need to evict:")
						peek_item = client_curr_cache_heap.peekitem()
						#print(f"client_mapped_id: {client_mapped_id}, peek_item: {peek_item}, curr_item: {index}")

						peek_item_weight = peek_item[1][0]
						peek_item_freq = peek_item[1][1]
						# print(f"does item exist in ghost map yet?")
						# if str(index) in client_ghost_cache_heap:
						#     print(f"client_mapped_id: {client_mapped_id}, yes {str(index)} exists in client_ghost_cache_heap")
						# else:
						#     print(f"client_mapped_id: {client_mapped_id}, no {str(index)} does not exist in client_ghost_cache_heap")
						# print(f"client_mapped_id: {client_mapped_id}, curr_item_weights: {client_ghost_cache_heap[str(index)]}")
						curr_item_weight = client_ghost_cache_heap[str(index)][0]
						curr_item_freq = client_ghost_cache_heap[str(index)][1]

						#print(f"client_mapped_id: {client_mapped_id}, peek_item_weight: {peek_item_weight}, type: {type(peek_item_weight)} peek_item_freq: {peek_item_freq}, type: {type(peek_item_freq)} curr_item_weight: {curr_item_weight}, type: {type(curr_item_weight)} , curr_item_freq:{curr_item_freq}, type: {type(curr_item_freq)}")
						peek_item_weight = float(peek_item_weight)
						peek_item_freq = float(peek_item_freq)
						curr_item_weight = float(curr_item_weight)
						curr_item_freq = float(curr_item_freq)
						#print(f"just before calc: client_mapped_id: {client_mapped_id}, maxw: {maxw},type: {type(maxw)} minw:{minw}, type:{type(minw)}, maxf:{maxf}, type: {type(maxf)} minf: {minf}, type: {type(minf)}")
						if (maxw - minw) > 0 and (maxf - minf) > 0:
							peek_item_experience_weight = (peek_item_weight - minw) / (maxw - minw)
							#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience_weight: {peek_item_experience_weight}, type: {type(peek_item_experience_weight)}")
							#peek_item_experience_freq =  (maxf - peek_item_freq ) / (maxf - minf)
							peek_item_experience_freq =  (peek_item_freq - minf) / (maxf - minf)
							#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience_freq: {peek_item_experience_freq}, type: {type(peek_item_experience_freq)}")
							curr_item_experience_weight = (curr_item_weight - minw) / (maxw - minw)
							#print(f"client_mapped_id: {client_mapped_id}, curr_item_experience_weight: {curr_item_experience_weight}, type: {type(curr_item_experience_weight)}")
							#curr_item_experience_freq =  (maxf - curr_item_freq ) / (maxf - minf)
							curr_item_experience_freq =  (curr_item_freq - minf) / (maxf - minf)
							#print(f"client_mapped_id: {client_mapped_id}, curr_item_experience_freq: {curr_item_experience_freq}, type: {type(curr_item_experience_freq)}")
							peek_item_experience = peek_item_experience_weight + peek_item_experience_freq
							curr_item_experience = curr_item_experience_weight + curr_item_experience_freq
						else:
							peek_item_experience = peek_item
							curr_item_experience = client_ghost_cache_heap[str(index)]

						#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience: {peek_item_experience}, curr_item_experience: {curr_item_experience}, type: {type(peek_item_experience)}, type: {type(curr_item_experience)}")
						#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience: {peek_item_experience}, curr_item_experience: {curr_item_experience}")
						# peek_item_experience = peek_item_experience_weight + peek_item_experience_freq
						# curr_item_experience = curr_item_experience_weight + curr_item_experience_freq

						#print(f"client_ghost_cache_heap {client_ghost_cache_heap}")
						# for key, value in client_ghost_cache_heap.items():
						#     print(f" client_ghost_cache_heap: K:{key}, V:{value}")
						#print(f"client_ghost_cache_heap[{index}] : {client_ghost_cache_heap[index]} , {peek_item}")
						#print(f"peek_item {peek_item}, peek_item[1]: {peek_item[1]}, type: {type(peek_item[1])}")
						#print(f"index {index}, type: {type(index)}")
						#print(f"index: {index}, imp val: {client_ghost_cache_heap[str(index)]}, type: {type(client_ghost_cache_heap[str(index)])}")
						#print(f"curr len of client_curr_cache_heap: {len(client_curr_cache_heap)}")
						if client_ghost_cache_heap[str(index)] > peek_item[1]: 
						#if curr_item_experience > peek_item_experience:
							evicted_item = client_curr_cache_heap.popitem()
							#print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))
							#print(f"evicted item {evicted_item}")
							#print(f"Evicting index: {evicted_item[0]} Weight: {evicted_item[1]}")# %(evicted_item[0], evicted_item[1]))
							print(f"client_mapped_id: {client_mapped_id}, Inserting Index: {index}, value: {client_ghost_cache_heap[str(index)]}, curr_experience: {curr_item_experience}")
							print(f"client_mapped_id: {client_mapped_id}, Evicting index: {evicted_item[0]} Weight: {evicted_item[1][0]} Frequency: {evicted_item[1][1]}, evict_experience: {peek_item_experience}")

							#print(f"before removing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")
							client_cur_cache_samples.remove(str(evicted_item[0]))
							#print(f"after removing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")

							self.write_list_to_csv(client_cur_cache_samples, cur_cache_path)

							# client_cur_cache_samples = self.read_list_from_csv(cur_cache_path)
							# print(f"after removing and writing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")

							key_id_map_ind = client_mapped_id + '_'+ str(evicted_item[0])

							# prefix = client_mapped_id + '_'
							# all_keys = self.key_id_map.keys()
							# all_keys_string = [key.decode('ascii') for key in all_keys]
							# matching_keys = [key for key in all_keys_string if key.startswith(prefix)]
							# print(f"before removing from redis. clientID: {client_mapped_id}, len(matching_keys): {len(matching_keys)}, keys: {matching_keys}")
							if self.key_id_map.exists(key_id_map_ind):
								#print(f"deleting {key_id_map_ind} from client_mapped_id: {client_mapped_id}")
								self.key_id_map.delete(key_id_map_ind)
								# if self.key_id_map.exists(key_id_map_ind):
								#     print(f"{key_id_map_ind} from client_mapped_id: {client_mapped_id} not deleted.")
								# else:
								#     print(f"{key_id_map_ind} from client_mapped_id: {client_mapped_id} deleted Successfully")

							#print(f"deleted {key_id_map_ind}")
							# all_keys = self.key_id_map.keys()
							# all_keys_string = [key.decode('ascii') for key in all_keys]
							# matching_keys = [key for key in all_keys_string if key.startswith(prefix)]
							# print(f"after removing from redis. clientID: {client_mapped_id}, len(matching_keys): {len(matching_keys)}, keys: {matching_keys}")
							keys_cnt-=1

					except:
						#print("Could not evict item or PQ was empty or PQ did not have that item.")
						pass

				if(keys_cnt >= self.cache_portion and self.barrier == 0.0 and self.policy == 'fedcaseimp'):
					try:
						#print("Check if we need to evict:")
						peek_item = client_curr_cache_heap.peekitem()
						#print(f"client_mapped_id: {client_mapped_id}, peek_item: {peek_item}, curr_item: {index}")

						peek_item_weight = peek_item[1][0]
						peek_item_freq = peek_item[1][1]
						# print(f"does item exist in ghost map yet?")
						# if str(index) in client_ghost_cache_heap:
						#     print(f"client_mapped_id: {client_mapped_id}, yes {str(index)} exists in client_ghost_cache_heap")
						# else:
						#     print(f"client_mapped_id: {client_mapped_id}, no {str(index)} does not exist in client_ghost_cache_heap")
						# print(f"client_mapped_id: {client_mapped_id}, curr_item_weights: {client_ghost_cache_heap[str(index)]}")
						curr_item_weight = client_ghost_cache_heap[str(index)][0]
						curr_item_freq = client_ghost_cache_heap[str(index)][1]

						#print(f"client_mapped_id: {client_mapped_id}, peek_item_weight: {peek_item_weight}, type: {type(peek_item_weight)} peek_item_freq: {peek_item_freq}, type: {type(peek_item_freq)} curr_item_weight: {curr_item_weight}, type: {type(curr_item_weight)} , curr_item_freq:{curr_item_freq}, type: {type(curr_item_freq)}")
						peek_item_weight = float(peek_item_weight)
						peek_item_freq = float(peek_item_freq)
						curr_item_weight = float(curr_item_weight)
						curr_item_freq = float(curr_item_freq)
						#print(f"just before calc: client_mapped_id: {client_mapped_id}, maxw: {maxw},type: {type(maxw)} minw:{minw}, type:{type(minw)}, maxf:{maxf}, type: {type(maxf)} minf: {minf}, type: {type(minf)}")
						if (maxw - minw) > 0 and (maxf - minf) > 0:
							peek_item_experience_weight = (peek_item_weight - minw) / (maxw - minw)
							#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience_weight: {peek_item_experience_weight}, type: {type(peek_item_experience_weight)}")
							#peek_item_experience_freq =  (maxf - peek_item_freq ) / (maxf - minf)
							peek_item_experience_freq =  (peek_item_freq - minf) / (maxf - minf)
							#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience_freq: {peek_item_experience_freq}, type: {type(peek_item_experience_freq)}")
							curr_item_experience_weight = (curr_item_weight - minw) / (maxw - minw)
							#print(f"client_mapped_id: {client_mapped_id}, curr_item_experience_weight: {curr_item_experience_weight}, type: {type(curr_item_experience_weight)}")
							#curr_item_experience_freq =  (maxf - curr_item_freq ) / (maxf - minf)
							curr_item_experience_freq =  (curr_item_freq - minf) / (maxf - minf)
							#print(f"client_mapped_id: {client_mapped_id}, curr_item_experience_freq: {curr_item_experience_freq}, type: {type(curr_item_experience_freq)}")
							peek_item_experience = peek_item_experience_weight + peek_item_experience_freq
							curr_item_experience = curr_item_experience_weight + curr_item_experience_freq
						else:
							peek_item_experience = peek_item
							curr_item_experience = client_ghost_cache_heap[str(index)]

						#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience: {peek_item_experience}, curr_item_experience: {curr_item_experience}, type: {type(peek_item_experience)}, type: {type(curr_item_experience)}")
						#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience: {peek_item_experience}, curr_item_experience: {curr_item_experience}")
						# peek_item_experience = peek_item_experience_weight + peek_item_experience_freq
						# curr_item_experience = curr_item_experience_weight + curr_item_experience_freq

						#print(f"client_ghost_cache_heap {client_ghost_cache_heap}")
						# for key, value in client_ghost_cache_heap.items():
						#     print(f" client_ghost_cache_heap: K:{key}, V:{value}")
						#print(f"client_ghost_cache_heap[{index}] : {client_ghost_cache_heap[index]} , {peek_item}")
						#print(f"peek_item {peek_item}, peek_item[1]: {peek_item[1]}, type: {type(peek_item[1])}")
						#print(f"index {index}, type: {type(index)}")
						#print(f"index: {index}, imp val: {client_ghost_cache_heap[str(index)]}, type: {type(client_ghost_cache_heap[str(index)])}")
						#print(f"curr len of client_curr_cache_heap: {len(client_curr_cache_heap)}")
						if client_ghost_cache_heap[str(index)] > peek_item[1]: 
						#if curr_item_experience > peek_item_experience:
							evicted_item = client_curr_cache_heap.popitem()
							#print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))
							#print(f"evicted item {evicted_item}")
							#print(f"Evicting index: {evicted_item[0]} Weight: {evicted_item[1]}")# %(evicted_item[0], evicted_item[1]))
							print(f"client_mapped_id: {client_mapped_id}, Inserting Index: {index}, value: {client_ghost_cache_heap[str(index)]}, curr_experience: {curr_item_experience}")
							print(f"client_mapped_id: {client_mapped_id}, Evicting index: {evicted_item[0]} Weight: {evicted_item[1][0]} Frequency: {evicted_item[1][1]}, evict_experience: {peek_item_experience}")

							#print(f"before removing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")
							client_cur_cache_samples.remove(str(evicted_item[0]))
							#print(f"after removing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")

							self.write_list_to_csv(client_cur_cache_samples, cur_cache_path)

							# client_cur_cache_samples = self.read_list_from_csv(cur_cache_path)
							# print(f"after removing and writing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")

							key_id_map_ind = client_mapped_id + '_'+ str(evicted_item[0])

							# prefix = client_mapped_id + '_'
							# all_keys = self.key_id_map.keys()
							# all_keys_string = [key.decode('ascii') for key in all_keys]
							# matching_keys = [key for key in all_keys_string if key.startswith(prefix)]
							# print(f"before removing from redis. clientID: {client_mapped_id}, len(matching_keys): {len(matching_keys)}, keys: {matching_keys}")
							if self.key_id_map.exists(key_id_map_ind):
								#print(f"deleting {key_id_map_ind} from client_mapped_id: {client_mapped_id}")
								self.key_id_map.delete(key_id_map_ind)
								# if self.key_id_map.exists(key_id_map_ind):
								#     print(f"{key_id_map_ind} from client_mapped_id: {client_mapped_id} not deleted.")
								# else:
								#     print(f"{key_id_map_ind} from client_mapped_id: {client_mapped_id} deleted Successfully")

							#print(f"deleted {key_id_map_ind}")
							# all_keys = self.key_id_map.keys()
							# all_keys_string = [key.decode('ascii') for key in all_keys]
							# matching_keys = [key for key in all_keys_string if key.startswith(prefix)]
							# print(f"after removing from redis. clientID: {client_mapped_id}, len(matching_keys): {len(matching_keys)}, keys: {matching_keys}")
							keys_cnt-=1

					except:
						#print("Could not evict item or PQ was empty or PQ did not have that item.")
						pass

				if(keys_cnt >= self.cache_portion and self.barrier == 0.0 and self.policy == 'fedcase'):
					try:
						#print("Check if we need to evict:")
						peek_item = client_curr_cache_heap.peekitem()
						#print(f"client_mapped_id: {client_mapped_id}, peek_item: {peek_item}, curr_item: {index}")

						peek_item_weight = peek_item[1][0]
						peek_item_freq = peek_item[1][1]
						# print(f"does item exist in ghost map yet?")
						# if str(index) in client_ghost_cache_heap:
						#     print(f"client_mapped_id: {client_mapped_id}, yes {str(index)} exists in client_ghost_cache_heap")
						# else:
						#     print(f"client_mapped_id: {client_mapped_id}, no {str(index)} does not exist in client_ghost_cache_heap")
						# print(f"client_mapped_id: {client_mapped_id}, curr_item_weights: {client_ghost_cache_heap[str(index)]}")
						curr_item_weight = client_ghost_cache_heap[str(index)][0]
						curr_item_freq = client_ghost_cache_heap[str(index)][1]

						#print(f"client_mapped_id: {client_mapped_id}, peek_item_weight: {peek_item_weight}, type: {type(peek_item_weight)} peek_item_freq: {peek_item_freq}, type: {type(peek_item_freq)} curr_item_weight: {curr_item_weight}, type: {type(curr_item_weight)} , curr_item_freq:{curr_item_freq}, type: {type(curr_item_freq)}")
						peek_item_weight = float(peek_item_weight)
						peek_item_freq = float(peek_item_freq)
						curr_item_weight = float(curr_item_weight)
						curr_item_freq = float(curr_item_freq)
						#print(f"just before calc: client_mapped_id: {client_mapped_id}, maxw: {maxw},type: {type(maxw)} minw:{minw}, type:{type(minw)}, maxf:{maxf}, type: {type(maxf)} minf: {minf}, type: {type(minf)}")
						if (maxw - minw) > 0 and (maxf - minf) > 0:
							peek_item_experience_weight = (peek_item_weight - minw) / (maxw - minw)
							#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience_weight: {peek_item_experience_weight}, type: {type(peek_item_experience_weight)}")
							#peek_item_experience_freq =  (maxf - peek_item_freq ) / (maxf - minf)
							peek_item_experience_freq =  (peek_item_freq - minf) / (maxf - minf)
							#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience_freq: {peek_item_experience_freq}, type: {type(peek_item_experience_freq)}")
							curr_item_experience_weight = (curr_item_weight - minw) / (maxw - minw)
							#print(f"client_mapped_id: {client_mapped_id}, curr_item_experience_weight: {curr_item_experience_weight}, type: {type(curr_item_experience_weight)}")
							#curr_item_experience_freq =  (maxf - curr_item_freq ) / (maxf - minf)
							curr_item_experience_freq =  (curr_item_freq - minf) / (maxf - minf)
							#print(f"client_mapped_id: {client_mapped_id}, curr_item_experience_freq: {curr_item_experience_freq}, type: {type(curr_item_experience_freq)}")
							peek_item_experience = peek_item_experience_weight + peek_item_experience_freq
							curr_item_experience = curr_item_experience_weight + curr_item_experience_freq
						else:
							peek_item_experience = peek_item
							curr_item_experience = client_ghost_cache_heap[str(index)]

						#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience: {peek_item_experience}, curr_item_experience: {curr_item_experience}, type: {type(peek_item_experience)}, type: {type(curr_item_experience)}")
						#print(f"client_mapped_id: {client_mapped_id}, peek_item_experience: {peek_item_experience}, curr_item_experience: {curr_item_experience}")
						# peek_item_experience = peek_item_experience_weight + peek_item_experience_freq
						# curr_item_experience = curr_item_experience_weight + curr_item_experience_freq

						#print(f"client_ghost_cache_heap {client_ghost_cache_heap}")
						# for key, value in client_ghost_cache_heap.items():
						#     print(f" client_ghost_cache_heap: K:{key}, V:{value}")
						#print(f"client_ghost_cache_heap[{index}] : {client_ghost_cache_heap[index]} , {peek_item}")
						#print(f"peek_item {peek_item}, peek_item[1]: {peek_item[1]}, type: {type(peek_item[1])}")
						#print(f"index {index}, type: {type(index)}")
						#print(f"index: {index}, imp val: {client_ghost_cache_heap[str(index)]}, type: {type(client_ghost_cache_heap[str(index)])}")
						#print(f"curr len of client_curr_cache_heap: {len(client_curr_cache_heap)}")
						#if client_ghost_cache_heap[str(index)] > peek_item[1]: 
						if curr_item_experience > peek_item_experience:
							evicted_item = client_curr_cache_heap.popitem()
							#print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))
							#print(f"evicted item {evicted_item}")
							#print(f"Evicting index: {evicted_item[0]} Weight: {evicted_item[1]}")# %(evicted_item[0], evicted_item[1]))
							print(f"client_mapped_id: {client_mapped_id}, Inserting Index: {index}, value: {client_ghost_cache_heap[str(index)]}, curr_experience: {curr_item_experience}")
							print(f"client_mapped_id: {client_mapped_id}, Evicting index: {evicted_item[0]} Weight: {evicted_item[1][0]} Frequency: {evicted_item[1][1]}, evict_experience: {peek_item_experience}")

							#print(f"before removing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")
							client_cur_cache_samples.remove(str(evicted_item[0]))
							#print(f"after removing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")

							self.write_list_to_csv(client_cur_cache_samples, cur_cache_path)

							# client_cur_cache_samples = self.read_list_from_csv(cur_cache_path)
							# print(f"after removing and writing {str(evicted_item[0])} from client_mapped_id: {client_mapped_id}, client_cur_cache_samples: {client_cur_cache_samples}, len: {len(client_cur_cache_samples)}")

							key_id_map_ind = client_mapped_id + '_'+ str(evicted_item[0])

							# prefix = client_mapped_id + '_'
							# all_keys = self.key_id_map.keys()
							# all_keys_string = [key.decode('ascii') for key in all_keys]
							# matching_keys = [key for key in all_keys_string if key.startswith(prefix)]
							# print(f"before removing from redis. clientID: {client_mapped_id}, len(matching_keys): {len(matching_keys)}, keys: {matching_keys}")
							if self.key_id_map.exists(key_id_map_ind):
								#print(f"deleting {key_id_map_ind} from client_mapped_id: {client_mapped_id}")
								self.key_id_map.delete(key_id_map_ind)
								# if self.key_id_map.exists(key_id_map_ind):
								#     print(f"{key_id_map_ind} from client_mapped_id: {client_mapped_id} not deleted.")
								# else:
								#     print(f"{key_id_map_ind} from client_mapped_id: {client_mapped_id} deleted Successfully")

							#print(f"deleted {key_id_map_ind}")
							# all_keys = self.key_id_map.keys()
							# all_keys_string = [key.decode('ascii') for key in all_keys]
							# matching_keys = [key for key in all_keys_string if key.startswith(prefix)]
							# print(f"after removing from redis. clientID: {client_mapped_id}, len(matching_keys): {len(matching_keys)}, keys: {matching_keys}")
							keys_cnt-=1
					
					except:
						#print("Could not evict item or PQ was empty or PQ did not have that item.")
						pass


				# random_int = random.randint(0, 20) 
				# keys_cnt = 0
				#if random_int % 2 == 0:
				#temporary addition for evaluation of other caching policies
				if self.cache_data and keys_cnt < self.cache_portion and self.barrier == 0:
					byte_stream = io.BytesIO()
					img.save(byte_stream,format=img.format)
					byte_stream.seek(0)
					byte_image = byte_stream.read()
					cid = client_mapped_id + "_" + str(index)
					tup_val = {"imgbyte": byte_image, "timestamp": currtimestamp}
					self.key_id_map.hmset(client_mapped_id + "_" + str(index), tup_val)
					#print(f"inserting index {cid} inside cache.")
					return_tup = self.key_id_map.hgetall(cid)
					ret_byte_image = return_tup[b'imgbyte']

					# if byte_image == ret_byte_image:
					# 	print(f"cid: {cid}, byte_image and ret_byte_image are same.")

					#print(f"{client_mapped_id} hitting {index}")
					#self.key_id_map.set(client_mapped_id + "_" + str(index), byte_image)
					client_cur_cache_samples.append(str(index))
					self.write_list_to_csv(client_cur_cache_samples, cur_cache_path)
					#print("Index: ", index)
				img = img.convert('RGB')

		else:
			img = Image.open(os.path.join(self.root, imgName))
			client = self.clients[index]
			client = int(client)
			fetched_from_cache = 0
			#print(f"data_file: {self.data_file}, in test client: {client}, index: {index}")

		######################################
		#print(f'client {self.clients[index]} is accessing image {index}: {imgName} from cache: {fetched_from_cache}') 
		# avoid channel error
		#print("image before transform: ", img)
		if img.mode != 'RGB':
			img = img.convert('RGB')

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		#print("image after transform: ", img)

		#return img, target, index
		#return client, img, target, index
		#print(f"data_file: {self.data_file} client: {client} index: {index}, fetched_from_cache: {fetched_from_cache}")
		return img, target, index, client, fetched_from_cache
		#return img, target

	def __len__(self):
		return len(self.data)

	@property
	def raw_folder(self):
		return self.root

	@property
	def processed_folder(self):
		return self.root

	def _check_exists(self):
		return (os.path.exists(os.path.join(self.processed_folder,
											self.data_file)))

	def form_heapdict(self,client_curr_cache_dict):

		client_curr_cache_heap = heapdict(client_curr_cache_dict)
		return client_curr_cache_heap

	def form_mapping_of_client_keys(self, client_cache_key_ids, curr_map):
		mapped_keys = {key: curr_map[key] for key in client_cache_key_ids}

		return mapped_keys

	def write_list_to_csv(self, my_list, file_path):
		with open(file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(my_list)

	def read_list_from_csv(self, file_path):
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			my_list = next(reader)

		#my_list = [int(item) for item in my_list]
		return my_list

	def read_imp_mapping_from_csv(self, file_path):
		samptoimp_map = {}
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)  # Skip header 
			for row in reader:
				samp = str(row[0])
				imp = float(row[1])
				samptoimp_map[samp] = imp
		return samptoimp_map

	def read_imp_freq_mapping_from_csv(self, file_path):
		samptoimpfreq_map = {}
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)  # Skip header
			for row in reader:
				samp = str(row[0])
				tup = ast.literal_eval(row[1])
				imp = float(tup[0])
				freq = float(tup[1])
				samptoimpfreq_map[samp] = (imp,freq)
			return samptoimpfreq_map

	def read_client_mapping(self, file_path):
		realtoass_map = {}
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)  # Skip header 
			for row in reader:
				real = int(row[0])
				assigned = int(row[1])
				realtoass_map[real] = assigned
		return realtoass_map

	def client_keys_in_cache(self,prefix):
			# Initialize the cursor to start from 0

		# prefix = prefix.decode('utf-8')
		# cursor = b"0"

		#cursor = "0"
		#keys_count = 0

		# Initialize a list to store the matching keys
		#matching_keys = []

		#print(f"prefix is {prefix}")

		#while True:
			# Use the SCAN command with the MATCH option to find keys matching the prefix

		all_keys = self.key_id_map.keys()

		all_keys_string = [key.decode('ascii') for key in all_keys]

		matching_keys = [key for key in all_keys_string if key.startswith(prefix)]

		#matching_vals = [self.key_id]

		keys_count = len(matching_keys)

		return keys_count, matching_keys

		# nodes_dict = self.key_id_map.scan(cursor=cursor, match=prefix +"*")
		# nodes_dict = self.key_id_map.scan(cursor=cursor, match=prefix +"*")
		# for node in nodes_dict:
		#     cursor, keys = nodes_dict[node]
		#     converted_keys = [key.decode('ascii') for key in keys]
		#     matching_keys.extend(converted_keys)
		#     keys_count += len(converted_keys)

		# return keys_count, matching_keys


	def load_meta_data(self, path):
		#datas, labels = [], []
		clients, datas, labels = [], [], [] #adding clients list

		with open(path) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count != 0: #skip the first row as it has headers client_id,sample_path,label_name,label_id
					clients.append(row[0]) #appending clients in clients list
					datas.append(row[1]) # take the location as data ex. data/raw_data/by_write/hsf_0/f0388_26/u0388_26/u0388_26_00004.png
					labels.append(int(row[-1])) # take the target as label ex. 6
				line_count += 1

		#return datas, labels # return a list of datas and a list of labels.
		return clients, datas, labels# return a list of clients, datas and a list of labels.

	def load_file(self, path):

		# load meta file to get labels
		# datas, labels = self.load_meta_data(os.path.join(
		#     self.processed_folder, 'client_data_mapping', self.data_file+'.csv'))
		clients, datas, labels = self.load_meta_data(os.path.join(
			self.processed_folder, 'client_data_mapping', self.data_file+'.csv'))

		#return datas, labels
		return clients, datas, labels