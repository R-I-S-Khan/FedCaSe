import logging
import math
import pickle 
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import fedscale.core.logger.execution as logger
import fedscale.core.config_parser as parser
import csv
import numpy as np
import ast

from fedscale.core.execution.optimizers import ClientOptimizer
from fedscale.dataloaders.nlp import mask_tokens


class Client(object):
	"""Basic client component in Federated Learning"""

	def __init__(self, conf):
		self.optimizer = ClientOptimizer()
		self.init_task(conf)
		self.num_fetched_from_cache = 0
		self.num_tot_samples = 0
		self.num_fetched_from_ssd = 0
		self.all_indices = torch.tensor([]).to(device = conf.device)
		self.all_weights = torch.tensor([]).to(device = conf.device)
		self.batch_wts = []
		# self.clientpath = os.path.join(parser.args.log_path, "logs", parser.args.job_name,
		#               parser.args.time_stamp, 'client')
		self.clientpath = os.path.join('/home/cc', 'client')


		self.curr_map = {}
		self.ghost_map = {}
	
	def init_task(self, conf):
		if conf.task == "detection":
			self.im_data = Variable(torch.FloatTensor(1).cuda())
			self.im_info = Variable(torch.FloatTensor(1).cuda())
			self.num_boxes = Variable(torch.LongTensor(1).cuda())
			self.gt_boxes = Variable(torch.FloatTensor(1).cuda())

		self.epoch_train_loss = 1e-4
		self.completed_steps = 0
		self.loss_squre = 0

	def loss_decompose(self, output, target, conf):
	
		criterion = nn.CrossEntropyLoss(reduce = False).to(device=conf.device)
		item_loss = criterion(output,target)
		loss = item_loss.mean()
		# loss -> loss needed for training
		# item_loss -> loss in item granularity. Needed for important sampling.
		return loss, item_loss 

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

	def read_samples_mapping_from_csv(self, file_path):
		samptoimpfreq_map = {}
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)  # Skip header
			for row in reader:
				client = int(row[0])
				tup = ast.literal_eval(row[1])
				cache = float(tup[0])
				ssd = float(tup[1])
				clienttosamp_map[client] = (cache,ssd)
			return clienttosamp_map

	def update_imp_mapping_in_csv(self, new_mapping, file_path):
		# curr_mapping = read_imp_mapping_from_csv(file_path)
		# curr_mapping.update(new_mapping)

		with open(file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['sample', 'imp'])  # Write header
			writer.writerows(new_mapping.items())

	def update_imp_freq_mapping_in_csv(self, new_mapping, file_path):
		# curr_mapping = read_imp_mapping_from_csv(file_path)
		# curr_mapping.update(new_mapping)

		with open(file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['sample', 'imp', 'freq'])  # Write header
			writer.writerows(new_mapping.items())

	def update_samples_mapping_in_csv(self, new_mapping, file_path):
		# curr_mapping = read_imp_mapping_from_csv(file_path)
		# curr_mapping.update(new_mapping)

		with open(file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['client', 'cache', 'ssd'])  # Write header
			writer.writerows(new_mapping.items())

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
		# if os.path.exists(file_path):
		#     logging.info(f"filepath {file_path} exists after creation.")
		# if os.path.isfile(file_path):
		#     logging.info(f"file {file_path} exists after creation.")

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

	def save_samples_mapping_to_csv(self, l1, l2, l3, file_path):
		mapping = {key: (val1, val2) for key, val1, val2 in zip(l1, l2, l3)}

		with open(file_path, 'w', newline='') as csvfile:
			#logging.info(f"file is opened {file_path}")
			writer = csv.writer(csvfile)
			#logging.info(f"csv writer initialized.")
			writer.writerow(['client', 'cache', 'ssd'])  # Write header
			#logging.info(f"header row initialized.")
			writer.writerows(mapping.items())
			#logging.info(f"items mapped in filepath.")

	def read_param(self,file_path):
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)  # Skip header 
			for row in reader:
				prev_loss = float(row[0])
				ls_param = float(row[1])
				#samptoimp_map[samp] = imp
		return prev_loss, ls_param

	def write_param(self,prev_loss, ls_param, file_path):
		with open(file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['prev_loss', 'ls_param'])  # Write header
			writer.writerow([prev_loss, ls_param])


	def read_samples_map(self,file_path):
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)  # Skip header 
			for row in reader:
				total = float(row[0])
				cache = float(row[1])
				ssd = float(row[2])
				#samptoimp_map[samp] = imp
		return total, cache, ssd

	def write_samples_map(self,total, cache, ssd, file_path):
		with open(file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['total', 'cache', 'ssd'])  # Write header
			writer.writerow([total, cache, ssd])

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


	def find_indices_weights(self,indices, loss, conf):

		#print(f"find_indices_weights getting called")
		#after the loss is obtained, we sort the losses in ascending order (lower loss first) after training each batch
		sorted_loss_indices = torch.argsort(loss).to(device = conf.device)
		#print(f"sorted_loss_indices : {sorted_loss_indices}")
		#print(sorted_loss_indices) #what are those loss array indices? Ex. the index at the 4th location(20) had the lowest loss(1), so 4 is first, then index at 1st location(12) had lowest loss(2), so then comes 1.
		indices_corr_loss = indices[sorted_loss_indices].to(device = conf.device) #what are the actual indices corresponding to the ascending order of loss array?
		#print(f"indices_corr_loss : {indices_corr_loss}")
		#self.all_indices.to(device = conf.device)
		appended_tensor = torch.cat((self.all_indices,indices_corr_loss), dim=0).to(device = conf.device)
		#print(f"appended_tensor : {appended_tensor}")
		self.all_indices = appended_tensor #find out all the indices that the client has trained on so far.
		#print(f"inside func all_indices : {self.all_indices}")

		#print(f"self.batch_wts : {self.batch_wts}")
		#self.all_weights.to(device = conf.device)
		appended_weights =  torch.cat((self.all_weights,self.batch_wts), dim=0).to(device = conf.device)
		#print(f"appended_tensor : {appended_tensor}")
		self.all_weights = appended_weights # find out the corresponding loss of all of the data samples that the client has trained on so far.
		
		#print(f"inside func all_weights : {self.all_weights}")

		#return self.all_indices, self.all_weights

	def train(self, client_data, model, conf):

		clientId = conf.clientId
		logging.info(f"Start to train (CLIENT: {clientId}) with {len(client_data.dataset)} samples...")
		#logging.info(f"{client_data.dataset}")

		tokenizer, device = conf.tokenizer, conf.device

		model = model.to(device=device)
		model.train()

		trained_unique_samples = min(
			len(client_data.dataset), conf.local_steps * conf.batch_size)
		self.global_model = None


		for j in range(conf.batch_size):
			self.batch_wts.append(math.log(j+10))
		self.batch_wts = torch.tensor(self.batch_wts).to(device = conf.device)

		if conf.gradient_policy == 'fed-prox':
			# could be move to optimizer
			self.global_model = [param.data.clone() for param in model.parameters()]

		optimizer = self.get_optimizer(model, conf)
		criterion = self.get_criterion(conf)
		error_type = None

		# NOTE: If one may hope to run fixed number of epochs, instead of iterations, 
		# use `while self.completed_steps < conf.local_steps * len(client_data)` instead

		self.weightpath = os.path.join(self.clientpath, str(clientId) + '.csv')

		ls_param_path = os.path.join(self.clientpath, str(clientId) + '_param.csv')

		self.ghostcachepath = os.path.join(self.clientpath, str(clientId) + '_ghost.csv')

		self.samples_path = os.path.join(self.clientpath, str(clientId) + '_samples.csv')

		self.realcalls_path = os.path.join(self.clientpath, str(clientId) + '_realcalls.csv')

		if os.path.exists(self.weightpath) and os.path.isfile(self.weightpath):
			#self.curr_map = self.read_imp_mapping_from_csv(self.weightpath)
			self.curr_map = self.read_imp_freq_mapping_from_csv(self.weightpath)
		if os.path.exists(self.ghostcachepath) and os.path.isfile(self.ghostcachepath):
			#self.curr_map = self.read_imp_mapping_from_csv(self.weightpath)
			self.ghost_map = self.read_imp_freq_mapping_from_csv(self.ghostcachepath)
			#logging.info(f"before client {clientId} map, {self.curr_map}")
		# if os.path.exists(self.samples_path) and os.path.isfile(self.samples_path):
		#     #self.curr_map = self.read_imp_mapping_from_csv(self.weightpath)
		#     self.samples_map = self.read_samples_mapping_from_csv(self.samples_path)



		self.num_fetched_from_cache = 0
		self.num_tot_samples = 0
		self.loss_squre = 0
		while self.completed_steps < conf.local_steps:
			try:
				self.train_step(client_data, conf, model, optimizer, criterion, clientId)
			except Exception as ex:
				error_type = ex
				break

		state_dicts = model.state_dict()
		model_param = {p: state_dicts[p].data.cpu().numpy()
					   for p in state_dicts}
		results = {'clientId': clientId, 'moving_loss': self.epoch_train_loss,
				   'trained_size': self.completed_steps*conf.batch_size, 
				   'success': self.completed_steps == conf.local_steps}

		if os.path.exists(self.realcalls_path) and os.path.isfile(self.realcalls_path):
			realcalls = self.read_client_numcalls(self.realcalls_path)
		else:
			realcalls = 0

		realcalls+=1

		self.write_client_numcalls(str(clientId), realcalls, self.realcalls_path)

		#logging.info()

		if os.path.exists(ls_param_path) and os.path.isfile(ls_param_path):
			prev_loss, ls_param = self.read_param(ls_param_path)
			cur_loss = self.epoch_train_loss
			# assume normal learning curve
			ls_fac = np.exp((cur_loss - prev_loss) / prev_loss)
			ls_param = ls_param * ls_fac
			prev_loss = cur_loss
			self.write_param(prev_loss, ls_param, ls_param_path)
		else:
			ls_init_fac = 1e-2
			prev_loss = self.epoch_train_loss
			ls_param = prev_loss * ls_init_fac
			self.write_param(prev_loss, ls_param, ls_param_path)

		self.write_samples_map(self.num_tot_samples, self.num_fetched_from_cache, self.num_fetched_from_ssd, self.samples_path)


		if error_type is None:
			logging.info(f"Training of (CLIENT: {clientId}) completes, results:{results}, total_samples_trained: {self.num_tot_samples}, from_cache: {self.num_fetched_from_cache}, from_ssd: {self.num_fetched_from_ssd}")
		else:
			logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

		results['utility'] = math.sqrt(
			self.loss_squre)*float(trained_unique_samples)
		results['update_weight'] = model_param
		results['wall_duration'] = 0
		results['num_fetched_from_cache'] = self.num_fetched_from_cache
		# results['all_indices'] = self.all_indices
		# results['all_weights'] = self.all_weights

		results['all_indices'] = torch.tensor(list(self.curr_map.keys())).to(device = conf.device)
		results['all_weights'] = torch.tensor(list(self.curr_map.values())).to(device = conf.device)

		results['mapping'] = self.curr_map

		if os.path.exists(self.weightpath) and os.path.isfile(self.weightpath):
			#logging.info(f"updating client {clientId} map, {self.curr_map}")
			#self.update_imp_mapping_in_csv(self.curr_map, self.weightpath)
			self.update_imp_freq_mapping_in_csv(self.curr_map, self.weightpath)

		else:
			samplelist = self.all_indices.to(torch.int).tolist()
			weightlist = self.all_weights.tolist()
			freqlist = self.all_freqs.tolist()
			if not os.path.exists(self.clientpath):
				os.makedirs(self.clientpath)
			#logging.info(f"making client {clientId} map")
			#self.save_imp_mapping_to_csv(samplelist, weightlist, self.weightpath)
			self.save_imp_freq_mapping_to_csv(samplelist, weightlist, freqlist, self.weightpath)

		#print(f"all_indices {clientId}: {results['all_indices']}")
		#print(f"all_weights {clientId}: {results['all_weights']}")

		return results

	def get_optimizer(self, model, conf):
		optimizer = None
		if conf.task == "detection":
			lr = conf.learning_rate
			params = []
			for key, value in dict(model.named_parameters()).items():
				if value.requires_grad:
					if 'bias' in key:
						params += [{'params': [value], 'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1),
									'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
					else:
						params += [{'params': [value], 'lr':lr,
									'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
			optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

		elif conf.task == 'nlp':

			no_decay = ["bias", "LayerNorm.weight"]
			optimizer_grouped_parameters = [
				{
					"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
					"weight_decay": conf.weight_decay,
				},
				{
					"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
					"weight_decay": 0.0,
				},
			]
			# Bert pre-training setup
			optimizer = torch.optim.Adam(
				optimizer_grouped_parameters, lr=conf.learning_rate, weight_decay=1e-2)
		else:
			optimizer = torch.optim.SGD(
				model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)
		return optimizer

	def get_criterion(self, conf):

		criterion = None
		if conf.task == 'voice':
			from torch_baidu_ctc import CTCLoss
			criterion = CTCLoss(reduction='none').to(device=conf.device)
		else:
			criterion = torch.nn.CrossEntropyLoss(
				reduction='none').to(device=conf.device)
		return criterion

	def train_step(self, client_data, conf, model, optimizer, criterion, clientId):

		#print("client_data: ", client_data) #<torch.utils.data.dataloader.DataLoader object at 0x7f0e41b67350>
		#print("conf: ", conf) #all of the conf parameters including default ones.
		batch_no = 0

		for data_pair in client_data:
			if conf.task == 'nlp':
				logging.info(f"This is a nlp task.")
				(data, _) = data_pair
				data, target = mask_tokens(
					data, tokenizer, conf, device=conf.device)
			elif conf.task == 'voice':
				logging.info(f"This is a voice recognition task.")
				(data, target, input_percentages,
					target_sizes), _ = data_pair
				input_sizes = input_percentages.mul_(
					int(data.size(3))).int()
			elif conf.task == 'detection':
				logging.info(f"This is a detection task.")
				temp_data = data_pair
				target = temp_data[4]
				data = temp_data[0:4]
			else:
				#(data, target) = data_pair
				#(data, target, index) = data_pair
				#(client, data, target, index) = data_pair #extracting client id from data_pair
				(data, target, index, client, fetched_from_cache) = data_pair #extracting client id from data_pair

			#print("type(data): ", type(data)) #tensor of samples in a batch 
			#print("len(data): ", len(data)) #number of samples in a batch

			#print("indexes: ", index) #indexes of those samples
			#print("type(index): ", type(index)) #tensor
			#print("type(client): ", type(client))

			index_list = index.tolist()
			fetched_from_cache = fetched_from_cache.tolist()

			self.num_tot_samples += len(index_list)
			#logging.info(f"index list {index_list}")
			#client = client.tolist()
			#logging.info(f"training client: {clientId}, client of samples {client}, index list {index_list}, type(index_list[0]): {type(index_list[0])}")
			#logging.info(f"training client: {clientId}, client of samples {client}, index list {index_list}, fetched_from_cache: {fetched_from_cache}")
			#logging.info(f"batch_no: {batch_no} len(data): {len(data)} , len(index_list): {len(index_list)}")
			#logging.info(f"batch_no: {batch_no} self.num_fetched_from_cache before: {self.num_fetched_from_cache}")
			#self.num_fetched_from_cache += len(index_list)//2
			self.num_fetched_from_cache += sum(fetched_from_cache)
			self.num_fetched_from_ssd = self.num_tot_samples - self.num_fetched_from_cache
			#logging.info(f"batch_no: {batch_no} self.num_fetched_from_cache after: {self.num_fetched_from_cache}")


			if conf.task == "detection":
				self.im_data.resize_(data[0].size()).copy_(data[0])
				self.im_info.resize_(data[1].size()).copy_(data[1])
				self.gt_boxes.resize_(data[2].size()).copy_(data[2])
				self.num_boxes.resize_(data[3].size()).copy_(data[3])
			elif conf.task == 'speech':
				data = torch.unsqueeze(data, 1).to(device=conf.device)
			elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
				(data, masks) = data
				data, masks = Variable(data).to(
					device=conf.device), Variable(masks).to(device=conf.device)

			else:
				data = Variable(data).to(device=conf.device)

			target = Variable(target).to(device=conf.device)
			index = Variable(index).to(device=conf.device)
			client = Variable(client).to(device=conf.device) #putting client id to device

			if conf.task == 'nlp':
				outputs = model(data, labels=target)
				loss = outputs[0]
			elif conf.task == 'voice':
				outputs, output_sizes = model(data, input_sizes)
				outputs = outputs.transpose(0, 1).float()  # TxNxH
				loss = criterion(
					outputs, target, output_sizes, target_sizes)
			elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
				outputs = model(
					data, attention_mask=masks, labels=target)
				loss = outputs.loss
				output = outputs.logits
			elif conf.task == "detection":
				rois, cls_prob, bbox_pred, \
					rpn_loss_cls, rpn_loss_box, \
					RCNN_loss_cls, RCNN_loss_bbox, \
					rois_label = model(
						self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

				loss = rpn_loss_cls + rpn_loss_box \
					+ RCNN_loss_cls + RCNN_loss_bbox

				loss_rpn_cls = rpn_loss_cls.item()
				loss_rpn_box = rpn_loss_box.item()
				loss_rcnn_cls = RCNN_loss_cls.item()
				loss_rcnn_box = RCNN_loss_bbox.item()
				
			else:
				output = model(data)
				loss = criterion(output, target)
				#print(f"inside {clientId}: output:{output}, epoch_train_loss:{target}, loss: {loss}")

				#logging.info(f"training client: {clientId}, loss: {loss}")

				#loss, item_loss = self.loss_decompose(output,target,conf) #decompose loss

			self.find_indices_weights(index,loss, conf)
			#print(f"inside trainstep all_indices {clientId}: {self.all_indices}")
			#print(f"inside trainstep all_weights {clientId}: {self.all_weights}")

			samples = self.all_indices.to(torch.int).tolist()
			weights = self.all_weights.tolist()
			new_freqs = []
			for sample in samples:
				if sample in self.curr_map:
					value = self.curr_map[sample]
					freqsample = value[1] + 1.0
					new_freqs.append(freqsample)
				else:
					new_freqs.append(1.0)

			# for key in y:
			#     if key in x:
			#         value = x[key]
			#         x[key] = (value[0], value[1] + 1)

			samptoimpfreq_map = {key: (val1, val2) for key, val1, val2 in zip(samples, weights, new_freqs)}
			#samptoimp_map = dict(zip(samples,weights))
			#self.curr_map.update(samptoimp_map)
			self.curr_map.update(samptoimpfreq_map)

			if os.path.exists(self.weightpath) and os.path.isfile(self.weightpath):
			#logging.info(f"updating client {clientId} map, {self.curr_map}")
				#self.update_imp_mapping_in_csv(self.curr_map, self.weightpath)
				self.update_imp_freq_mapping_in_csv(self.curr_map, self.weightpath)

			if os.path.exists(self.ghostcachepath) and os.path.isfile(self.ghostcachepath):
				#logging.info(f"updating client {clientId} map, {self.curr_map}")
				self.ghost_map.update(samptoimpfreq_map)
				self.update_imp_freq_mapping_in_csv(self.ghost_map, self.ghostcachepath)
			else:
				# samplelist = self.all_indices.to(torch.int).tolist()
				# weightlist = self.all_weights.tolist()
				# if not os.path.exists(self.ghostcachepath):
				#     os.makedirs(self.ghostcachepath)
				#logging.info(f"making client {clientId} map")
				self.save_imp_freq_mapping_to_csv(samples, weights, new_freqs, self.ghostcachepath)

			# ======== collect training feedback for other decision components [e.g., oort selector] ======

			if conf.task == 'nlp' or (conf.task == 'text_clf' and conf.model == 'albert-base-v2'):
				loss_list = [loss.item()]  # [loss.mean().data.item()]

			elif conf.task == "detection":
				loss_list = [loss.tolist()]
				loss = loss.mean()
			else:
				loss_list = loss.tolist()
				loss = loss.mean()

			temp_loss = sum(loss_list)/float(len(loss_list))
			self.loss_squre += sum([l**2 for l in loss_list]
								)/float(len(loss_list))
			# only measure the loss of the first epoch
			if self.completed_steps < len(client_data):
				if self.epoch_train_loss == 1e-4:
					self.epoch_train_loss = temp_loss
				else:
					self.epoch_train_loss = (
						1. - conf.loss_decay) * self.epoch_train_loss + conf.loss_decay * temp_loss

			# ========= Define the backward loss ==============
			#print(f"inside trainstep all_indices {clientId}: allloss:{loss_list}, epoch_train_loss:{self.epoch_train_loss}")
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# ========= Weight handler ========================
			self.optimizer.update_client_weight(
				conf, model, self.global_model if self.global_model is not None else None)

			self.completed_steps += 1

			if self.completed_steps == conf.local_steps:
				break
			batch_no+=1
		#logging.info(f"batch_no: {batch_no} client_data_len: {len(client_data.dataset)} , num_fetched_from_cache: {self.num_fetched_from_cache}")



	def test(self, conf):
		pass
