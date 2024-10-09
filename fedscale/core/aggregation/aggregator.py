# -*- coding: utf-8 -*-

import pickle
import threading
from concurrent import futures

import grpc
import torch
from torch.utils.tensorboard import SummaryWriter

import fedscale.core.channels.job_api_pb2_grpc as job_api_pb2_grpc
import fedscale.core.logger.aggragation as logger
import fedscale.core.config_parser as parser
from fedscale.core import commons
from fedscale.core.channels import job_api_pb2
from fedscale.core.resource_manager import ResourceManager
from fedscale.core.fllibs import *
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
import statistics

MAX_MESSAGE_LENGTH = 1*1024*1024*1024  # 1GB


class Aggregator(job_api_pb2_grpc.JobServiceServicer):
	"""This centralized aggregator collects training/testing feedbacks from executors
	
	Args:
		args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

	"""
	def __init__(self, args):
		# init aggregator loger
		logger.initiate_aggregator_setting()

		logging.info(f"Job args {args}")

		self.args = args
		self.experiment_mode = args.experiment_mode
		self.device = args.cuda_device if args.use_cuda else torch.device(
			'cpu')

		# ======== env information ========
		self.this_rank = 0
		self.global_virtual_clock = 0.
		self.round_duration = 0.
		self.resource_manager = ResourceManager(self.experiment_mode)
		self.client_manager = self.init_client_manager(args=args)

		# ======== model and data ========
		self.model = None
		self.model_in_update = 0
		self.update_lock = threading.Lock()
		# all weights including bias/#_batch_tracked (e.g., state_dict)
		self.model_weights = collections.OrderedDict()
		self.last_gradient_weights = []  # only gradient variables
		self.model_state_dict = None
		# NOTE: if <param_name, param_tensor> (e.g., model.parameters() in PyTorch), then False
		# True, if <param_name, list_param_tensors> (e.g., layer.get_weights() in Tensorflow)
		self.using_group_params = self.args.engine == commons.TENSORFLOW

		# ======== channels ========
		self.connection_timeout = self.args.connection_timeout
		self.executors = None
		self.grpc_server = None

		# ======== Event Queue =======
		self.individual_client_events = {}    # Unicast
		self.sever_events_queue = collections.deque()
		self.broadcast_events_queue = collections.deque()  # Broadcast

		# ======== runtime information ========
		self.tasks_round = 0
		self.num_of_clients = 0

		# NOTE: sampled_participants = sampled_executors in deployment,
		# because every participant is an executor. However, in simulation mode,
		# executors is the physical machines (VMs), thus:
		# |sampled_executors| << |sampled_participants| as an VM may run multiple participants
		self.sampled_participants = []
		self.sampled_executors = []

		self.round_stragglers = []
		self.model_update_size = 0.

		self.collate_fn = None
		self.task = args.task
		self.round = 0

		self.start_run_time = time.time()
		self.client_conf = {}

		self.stats_util_accumulator = []
		self.loss_accumulator = []
		self.client_training_results = []
		self.fedcase_clients = {}

		# number of registered executors
		self.registered_executor_info = set()
		self.test_result_accumulator = []
		self.testing_history = {'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
								'gradient_policy': args.gradient_policy, 'task': args.task, 'perf': collections.OrderedDict()}

		self.log_writer = SummaryWriter(log_dir=logger.logDir)
		self.clientpath = os.path.join('/home/cc', 'client')
		self.clientutilpath = os.path.join(self.clientpath,'clientutils')
		self.mintime = 99999
		self.maxtime = -99999
		self.minscore = 99999
		self.maxscore = -99999
		self.time_diff_bet_rounds = []
		self.loss_diff_bet_rounds = []
		self.prev_ratios = []

		self.time_diff_bet_rounds_min = 99999
		self.std_dev_time_diff = 99999
		self.loss_diff_bet_rounds_max = -99999
		self.std_dev_loss_diff = 99999
		self.aim_loss_diff_btwn_rounds = 99999
		self.aim_time_diff_btwn_rounds = 99999
		self.aim_loss_rate = 1.2
		self.prediction_ratio_prev = 0
		self.prev_round_loss = -1
		self.ratio_prev = 0
		self.random_participants = []


		# ======== Task specific ============
		self.init_task_context()


	def predict(self,rf_model, aim_loss_diff_btwn_rounds, aim_time_diff_btwn_rounds, aim_loss_rate):
		new_point = np.array([[aim_loss_diff_btwn_rounds, aim_time_diff_btwn_rounds, aim_loss_rate]])
		prediction = rf_model.predict(new_point)
		return prediction[0]

	def model_train(self,loss_diff_bet_rounds, time_diff_bet_rounds, prev_ratios):
		loss_diff_array= np.array(loss_diff_bet_rounds).reshape(-1, 1)
		#print(f"loss_diff_array: {loss_diff_array}")
		time_diff_array = np.array(time_diff_bet_rounds).reshape(-1, 1)
		#print(f"time_diff_array: {time_diff_array}")
		rate_loss_diff_array = loss_diff_array/time_diff_array
		prev_ratio_array= np.array(prev_ratios)
		rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
		rf_model.fit(np.hstack((loss_diff_array, time_diff_array, rate_loss_diff_array)), prev_ratio_array)
		return rf_model

	def setup_env(self):
		"""Set up experiments environment and server optimizer
		"""
		self.setup_seed(seed=1)
		self.optimizer = ServerOptimizer(
			self.args.gradient_policy, self.args, self.device)

	def setup_seed(self, seed=1):
		"""Set global random seed for better reproducibility

		Args:
			seed (int): random seed

		"""
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)
		torch.backends.cudnn.deterministic = True

	def init_control_communication(self):
		"""Create communication channel between coordinator and executor.
		This channel serves control messages.
		"""
		logging.info(f"Initiating control plane communication ...")
		if self.experiment_mode == commons.SIMULATION_MODE:
			num_of_executors = 0
			for ip_numgpu in self.args.executor_configs.split("="):
				ip, numgpu = ip_numgpu.split(':')
				for numexe in numgpu.strip()[1:-1].split(','):
					for _ in range(int(numexe.strip())):
						num_of_executors += 1
			self.executors = list(range(num_of_executors))
		else:
			self.executors = list(range(self.args.num_participants))

		# initiate a server process
		self.grpc_server = grpc.server(
			futures.ThreadPoolExecutor(max_workers=20),
			options=[
				('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
				('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
			],
		)
		job_api_pb2_grpc.add_JobServiceServicer_to_server(
			self, self.grpc_server)
		port = '[::]:{}'.format(self.args.ps_port)

		logging.info(f'%%%%%%%%%% Opening aggregator sever using port {port} %%%%%%%%%%')

		self.grpc_server.add_insecure_port(port)
		self.grpc_server.start()

	def init_data_communication(self):
		"""For jumbo traffics (e.g., training results).
		"""
		pass

	def init_model(self):
		"""Load the model architecture
		"""
		assert self.args.engine == commons.PYTORCH, "Please define model for non-PyTorch models"

		self.model = init_model()

		# Initiate model parameters dictionary <param_name, param>
		self.model_weights = self.model.state_dict()

	def init_task_context(self):
		"""Initiate execution context for specific tasks
		"""
		if self.args.task == "detection":
			cfg_from_file(self.args.cfg_file)
			np.random.seed(self.cfg.RNG_SEED)
			self.imdb, _, _, _ = combined_roidb(
				"voc_2007_test", ['DATA_DIR', self.args.data_dir], server=True)

	def init_client_manager(self, args):
		""" Initialize client sampler

		Args:
			args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
		
		Returns:
			clientManager: The client manager class

		Currently we implement two client managers:

		1. Random client sampler - it selects participants randomly in each round 
		[Ref]: https://arxiv.org/abs/1902.01046

		2. Oort sampler
		Oort prioritizes the use of those clients who have both data that offers the greatest utility
		in improving model accuracy and the capability to run training quickly.
		[Ref]: https://www.usenix.org/conference/osdi21/presentation/lai

		"""

		# sample_mode: random or oort
		client_manager = clientManager(args.sample_mode, args=args)

		return client_manager

	def load_client_profile(self, file_path):
		"""For Simulation Mode: load client profiles/traces

		Args:
			file_path (string): File path for the client profiles/traces

		Returns:
			dictionary: Return the client profiles/traces

		"""
		global_client_profile = {}
		if os.path.exists(file_path):
			with open(file_path, 'rb') as fin:
				# {clientId: [computer, bandwidth]}
				global_client_profile = pickle.load(fin)

		return global_client_profile

	def client_register_handler(self, executorId, info):
		"""Triggered once receive new executor registration.
		
		Args:
			executorId (int): Executor Id
			info (dictionary): Executor information

		"""
		logging.info(f"Loading {len(info['size'])} client traces ...")
		#logging.info(f"executorId: {executorId}, info: {info}")
		logging.info(f"executorId: {executorId}")
		# for key in info:
		#     logging.info(f'key: {key}')
		#     if key != 'size':
		#         logging.info(f'key content {info[key]}')

		for _size in info['size']: #for all of the clients prepared for this dataset
			#logging.info(f"_size is: {_size}")
			#logging.info("XXXXXXXXXXXXXXXXXX")
			#logging.info(f"info['size']:  {info['size']}") #these are the number of samples of each client info['size']:  [299, 151, 335, 157, 181, 354, 180, 179, ..]
			#logging.info("XXENDXX")
			# since the worker rankId starts from 1, we also configure the initial dataId as 1
			mapped_id = (self.num_of_clients+1) % len(
				self.client_profiles) if len(self.client_profiles) > 0 else 1
			# systemProfile = self.client_profiles.get(
			# 	mapped_id, {'computation': 1.0, 'communication': 1.0})
			systemProfile = self.client_profiles.get(
				mapped_id, {'computation': 1.0, 'communication': 1.0, 'sample_size': 100.0})

			clientId = (
				self.num_of_clients+1) if self.experiment_mode == commons.SIMULATION_MODE else executorId
			self.client_manager.register_client(
				executorId, clientId, size=_size, speed=systemProfile)
			self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
												 upload_step=self.args.local_steps, upload_size=self.model_update_size, download_size=self.model_update_size)
			self.num_of_clients += 1

		logging.info("Info of all feasible clients {}".format(
			self.client_manager.getDataInfo()))

	def executor_info_handler(self, executorId, info):
		"""Handler for register executor info and it will start the round after number of
		executor reaches requirement.
		
		Args:
			executorId (int): Executor Id
			info (dictionary): Executor information

		"""
		self.registered_executor_info.add(executorId)
		logging.info(f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}")

		# In this simulation, we run data split on each worker, so collecting info from one executor is enough
		# Waiting for data information from executors, or timeout
		if self.experiment_mode == commons.SIMULATION_MODE:

			if len(self.registered_executor_info) == len(self.executors):
				self.client_register_handler(executorId, info)
				# start to sample clients
				self.round_completion_handler()
		else:
			# In real deployments, we need to register for each client
			self.client_register_handler(executorId, info)
			if len(self.registered_executor_info) == len(self.executors):
				self.round_completion_handler()

	def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
		"""Record sampled client execution information in last round. In the SIMULATION_MODE,
		further filter the sampled_client and pick the top num_clients_to_collect clients.
		
		Args:
			sampled_clients (list of int): Sampled clients from client manager
			num_clients_to_collect (int): The number of clients actually needed for next round.

		Returns:
			tuple: Return the sampled clients and client execution information in the last round.

		"""
		if self.experiment_mode == commons.SIMULATION_MODE:
			# NOTE: We try to remove dummy events as much as possible in simulations,
			# by removing the stragglers/offline clients in overcommitment"""
			sampledClientsReal = []
			completionTimes = []
			completed_client_clock = {}
			previous_clients = []
			# org_sampledClientsReal  = []
			# org_completionTimes = []
			# org_completed_client_clock = {}
			#working with norm and ML model.
			
			if self.round >=10:
				#print(f"len(loss_diff): {len(self.loss_diff_bet_rounds)}, len(time_dff): {len(self.time_diff_bet_rounds)}, len(prev_ratios): {len(self.prev_ratios)}")
				rf_model = self.model_train(self.loss_diff_bet_rounds, self.time_diff_bet_rounds, self.prev_ratios)
				self.aim_loss_rate = self.aim_loss_diff_btwn_rounds / self.aim_time_diff_btwn_rounds
				self.prediction_ratio_prev = self.predict(rf_model, self.aim_loss_diff_btwn_rounds, self.aim_time_diff_btwn_rounds, self.aim_loss_rate)

			if len(self.fedcase_clients) > 0:
				sorted_clients_dict = dict(sorted(self.fedcase_clients.items(), key=lambda item: item[1], reverse=True))
				sorted_clients_list = list(sorted_clients_dict.keys())
				x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
				random_item = random.choice(x)
				self.ratio_prev = random_item

				# if (self.round + 1) >=3:
				# 	self.prev_ratios.append(ratio_prev)

				if self.prediction_ratio_prev != 0:
					self.ratio_prev = self.prediction_ratio_prev

				#self.ratio_prev = 0.1
				previous_clients_num = int(self.ratio_prev * num_clients_to_collect)
				previous_clients = sorted_clients_list[:previous_clients_num]
				prev_clients_set = set(previous_clients)
				logging.info(f"round: {self.round +1}, ratio_prev: {self.ratio_prev}, previous_clients_num: {previous_clients_num}, previous_clients: {previous_clients}")

			org_client_sampling = set(sampled_clients)
			sampled_clients = sampled_clients + previous_clients
			new_client_sampling = set(sampled_clients)
			common_elements = len(org_client_sampling.intersection(new_client_sampling))
			uncommon_elements = max(len(org_client_sampling), len(new_client_sampling)) - common_elements

			sampled_clients = set(sampled_clients)
			sampled_clients = list(sampled_clients)

			logging.info(f"sampled clients after: {sampled_clients}")
			logging.info(f"round: {self.round+1}, diff between client: {uncommon_elements}")

			# finding difference in client scheduling.
			# oort_clients = sampled_clients
			# fed_clients = self.random_participants[:num_clients_to_collect] + previous_clients

			# oort_clients_set = set(oort_clients)
			# fed_clients_set = set(fed_clients)
			# comm_oort_fed = len(oort_clients_set.intersection(fed_clients_set))
			# uncomm_oort_fed = max(len(oort_clients_set), len(fed_clients_set)) - comm_oort_fed
			# logging.info(f"round: {self.round + 1}, diff between different sched policies: {uncomm_oort_fed}")

			#temporary addition for evaluation of other caching policies
			# sampled_clients = org_client_sampling
			# sampled_clients = list(sampled_clients)
			#round_num = self.round + 1
			# if round_num % 10 == 0: #introduce randomness in training to introduce variety.
			# 	sampled_clients = org_client_sampling
			# 	sampled_clients = list(sampled_clients)


			count_client = 0
			check =1
			for client_to_run in sampled_clients:
				count_client+=1
				client_cfg = self.client_conf.get(client_to_run, self.args)

				exe_cost = self.client_manager.getCompletionTime(client_to_run,
																 batch_size=client_cfg.batch_size, upload_step=client_cfg.local_steps,
																 upload_size=self.model_update_size, download_size=self.model_update_size)

				roundDuration = exe_cost['computation'] + \
					exe_cost['communication'] + \
					exe_cost['io']
				# if the client is not active by the time of collection, we consider it is lost in this round
				if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
					sampledClientsReal.append(client_to_run)
					completionTimes.append(roundDuration)
					completed_client_clock[client_to_run] = exe_cost
		
			# ########### previous approach calculation ##############


			######################################################
			num_clients_to_collect = min(
				num_clients_to_collect, len(completionTimes))
			# 2. get the top-k completions to remove stragglers
			# these are the indices of the completion times
			sortedWorkersByCompletion = sorted(
				range(len(completionTimes)), key=lambda k: completionTimes[k])
			top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
			#map the completion indices to the actual clientIds, clientId indices and their corresponding completion time index is the same.
			clients_to_run = [sampledClientsReal[k] for k in top_k_index]
			#list of the slow active clients
			dummy_clients = [sampledClientsReal[k]
							 for k in sortedWorkersByCompletion[num_clients_to_collect:]]
			#get the completion time of the slowest node out of the selected active clients.
			round_duration = completionTimes[top_k_index[-1]]
			completionTimes.sort()

			run_clients_set = set(clients_to_run)
			if len(self.fedcase_clients) > 0:
				rand_fed_num = len(run_clients_set) - len(prev_clients_set)

				if rand_fed_num < 0:
					rand_fed_num = 0
				rand_fed = self.random_participants[:rand_fed_num]
				fedcase_clients = rand_fed + previous_clients

				#fedcase_clients = fedcase_clients[:num_clients_to_collect]

				fedcase_clients_set = set(fedcase_clients)

				common = run_clients_set & prev_clients_set
				count_common = len(common)
				count_uncommon = len(clients_to_run) - count_common
				count_uncommon_2 = len(previous_clients) - count_common
				logging.info(f"round: {self.round + 1}, sel_client_num: {len(clients_to_run)}, common_clients: {count_common}, uncommon_clients: {count_uncommon}, uncomm_clients2 :{count_uncommon_2}")

				common = run_clients_set & fedcase_clients_set
				count_common = len(common)
				count_uncommon = len(clients_to_run) - count_common
				logging.info(f"round: {self.round + 1}, len(clients_to_run): {len(clients_to_run)}, count_common: {count_common}, count_uncommon: {count_uncommon}")

			#client selection at this place.

			# clients_before_experience_based_selection = set(org_clients_to_run)
			# clients_after_experience_based_selection = set(clients_to_run)
			# common_clients = len(clients_after_experience_based_selection.intersection(clients_before_experience_based_selection))
			# uncommon_clients = max(len(clients_before_experience_based_selection),len(clients_after_experience_based_selection)) - common_clients
			# diff_in_round_time = int(org_round_duration) - int(round_duration)

			# logging.info(f"round: {self.round + 1}, difference between client sampling methods: {uncommon_clients}, diff_duration:{diff_in_round_time}")
			# logging.info(f"round: {self.round + 1}, org_clients_to_run: {org_clients_to_run}, clients_to_run:{clients_to_run}")			
			return (clients_to_run, dummy_clients,
					completed_client_clock, round_duration,
					completionTimes[:num_clients_to_collect])
		else:
			completed_client_clock = {
				client: {'computation': 1, 'communication': 1} for client in sampled_clients}
			completionTimes = [1 for c in sampled_clients]
			return (sampled_clients, sampled_clients, completed_client_clock,
					1, completionTimes)

	def run(self):
		"""Start running the aggregator server by setting up execution 
		and communication environment, and monitoring the grpc message.
		"""
		self.setup_env()
		self.init_control_communication()
		self.init_data_communication()

		self.init_model()
		self.save_last_param()
		self.model_update_size = sys.getsizeof(
			pickle.dumps(self.model))/1024.0*8.  # kbits
		self.client_profiles = self.load_client_profile(
			file_path=self.args.device_conf_file)

		self.event_monitor()

	def select_participants(self, select_num_participants, overcommitment=1.3):
		"""Select clients for next round.

		Args: 
			select_num_participants (int): Number of clients to select.
			overcommitment (float): Overcommit ration for next round.

		Returns:
			list of int: The list of sampled clients id.

		"""
		clients_for_policy, clients_random = self.client_manager.select_participants(
			int(select_num_participants*overcommitment),
			cur_time=self.global_virtual_clock)

		# return sorted(self.client_manager.select_participants(
		# 	int(select_num_participants*overcommitment),
		# 	cur_time=self.global_virtual_clock),
		# )
		need = int(select_num_participants*overcommitment)
		clients_passed = sorted(clients_for_policy,)
		clients_sel_random = sorted(clients_random,)
		logging.info(f"clients that should be selected: {need}, clients_passed: {len(clients_passed)}, clients_sel_random: {len(clients_sel_random)}")
		return sorted(clients_for_policy,),sorted(clients_random,)

	def client_completion_handler(self, results):
		"""We may need to keep all updates from clients, 
		if so, we need to append results to the cache
		
		Args:
			results (dictionary): client's training result
		
		"""
		# Format:
		#       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': round_train_loss,
		#       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

		if self.args.gradient_policy in ['q-fedavg']:
			self.client_training_results.append(results)
		# Feed metrics to client sampler
		self.stats_util_accumulator.append(results['utility'])
		self.loss_accumulator.append(results['moving_loss'])

		cur_client_duration = self.virtual_client_clock[results['clientId']]['computation']+ \
					  self.virtual_client_clock[results['clientId']]['communication']+ \
					  self.virtual_client_clock[results['clientId']]['io']
		cur_client_score = results['utility']

		self.mintime = min(self.mintime, cur_client_duration)
		self.maxtime = max(self.maxtime, cur_client_duration)
		self.minscore = min(self.minscore, results['utility'])
		self.maxscore = max(self.maxscore, results['utility'])

		if self.mintime == self.maxtime:
			normtime = 0
		else:
			normtime = (self.maxtime - cur_client_duration ) / (self.maxtime - self.mintime) #how much time could this client reduce in the entire spectrum

		if self.minscore == self.maxscore:
			normscore = 0
		else:
			normscore = (cur_client_score - self.minscore) / (self.maxscore - self.minscore) #how much important this client is within all of the clients trained so far.

		#weighted_client_comp_score = 0.6 * normtime + 0.4 * normscore
		weighted_client_comp_score = 0.1 * normtime + 0.9 * normscore
		#weighted_client_comp_score = normtime + normscore

		self.client_manager.register_feedback(results['clientId'], results['utility'],
										  auxi=math.sqrt(
											  results['moving_loss']),
										  time_stamp=self.round,
										  duration=self.virtual_client_clock[results['clientId']]['computation'] +
										  self.virtual_client_clock[results['clientId']]['communication']+
										  self.virtual_client_clock[results['clientId']]['io']
										  )

		self.fedcase_clients[results['clientId']] = weighted_client_comp_score

		#logging.info(f"res_clientid: {results['clientId']}, clients_recorded: {self.fedcase_clients}")

		# ================== Aggregate weights ======================
		self.update_lock.acquire()

		self.model_in_update += 1
		if self.using_group_params == True:
			self.aggregate_client_group_weights(results)
		else:
			self.aggregate_client_weights(results)

		self.update_lock.release()

	def aggregate_client_weights(self, results):
		"""May aggregate client updates on the fly
		
		Args:
			results (dictionary): client's training result
		
		[FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
		H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
		"""
		# Start to take the average of updates, and we do not keep updates to save memory
		# Importance of each update is 1/#_of_participants
		# importance = 1./self.tasks_round

		for p in results['update_weight']:
			param_weight = results['update_weight'][p]
			if isinstance(param_weight, list):
				param_weight = np.asarray(param_weight, dtype=np.float32)
			param_weight = torch.from_numpy(
				param_weight).to(device=self.device)

			if self.model_in_update == 1:
				self.model_weights[p].data = param_weight
			else:
				self.model_weights[p].data += param_weight

		if self.model_in_update == self.tasks_round:
			for p in self.model_weights:
				d_type = self.model_weights[p].data.dtype

				self.model_weights[p].data = (
					self.model_weights[p]/float(self.tasks_round)).to(dtype=d_type)

	def aggregate_client_group_weights(self, results):
		"""Streaming weight aggregation. Similar to aggregate_client_weights, 
		but each key corresponds to a group of weights (e.g., for Tensorflow)
		
		Args:
			results (dictionary): Client's training result
		
		"""
		for p_g in results['update_weight']:
			param_weights = results['update_weight'][p_g]
			for idx, param_weight in enumerate(param_weights):
				if isinstance(param_weight, list):
					param_weight = np.asarray(param_weight, dtype=np.float32)
				param_weight = torch.from_numpy(
					param_weight).to(device=self.device)

				if self.model_in_update == 1:
					self.model_weights[p_g][idx].data = param_weight
				else:
					self.model_weights[p_g][idx].data += param_weight

		if self.model_in_update == self.tasks_round:
			for p in self.model_weights:
				for idx in range(len(self.model_weights[p])):
					d_type = self.model_weights[p][idx].data.dtype

					self.model_weights[p][idx].data = (
						self.model_weights[p][idx].data/float(self.tasks_round)
					).to(dtype=d_type)

	def save_last_param(self):
		""" Save the last model parameters
		"""
		if self.args.engine == commons.TENSORFLOW:
			self.last_gradient_weights = [
				layer.get_weights() for layer in self.model.layers]
			self.model_weights = copy.deepcopy(self.model.state_dict())
		else:
			self.last_gradient_weights = [
				p.data.clone() for p in self.model.parameters()]
			self.model_weights = copy.deepcopy(self.model.state_dict())

	def update_default_task_config(self):
		"""Update the default task configuration after each round
		"""
		if self.round % self.args.decay_round == 0:
			self.args.learning_rate = max(
				self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

	def round_weight_handler(self, last_model):
		"""Update model when the round completes
		
		Args:
			last_model (list): A list of global model weight in last round.
		
		"""
		if self.round > 1:
			if self.args.engine == commons.TENSORFLOW:
				for layer in self.model.layers:
					layer.set_weights([p.cpu().detach().numpy()
									  for p in self.model_weights[layer.name]])
			else:
				self.model.load_state_dict(self.model_weights)
				current_grad_weights = [param.data.clone()
										for param in self.model.parameters()]
				self.optimizer.update_round_gradient(
					last_model, current_grad_weights, self.model)

	def round_completion_handler(self):
		"""Triggered upon the round completion, it registers the last round execution info,
		broadcast new tasks for executors and select clients for next round.
		"""
		#does this get triggered during testing? Are we calculating testing time in the wall clock time too?
		self.global_virtual_clock += self.round_duration
		self.round += 1

		# handle the global update w/ current and last
		self.round_weight_handler(self.last_gradient_weights)

		avgUtilLastround = sum(self.stats_util_accumulator) / \
			max(1, len(self.stats_util_accumulator))
		# assign avg reward to explored, but not ran workers
		for clientId in self.round_stragglers:
			self.client_manager.register_feedback(clientId, avgUtilLastround,
											  time_stamp=self.round,
											  duration=self.virtual_client_clock[clientId]['computation'] +
											  self.virtual_client_clock[clientId]['communication']+
											  self.virtual_client_clock[clientId]['io'],
											  success=False)

		avg_loss = sum(self.loss_accumulator) / \
			max(1, len(self.loss_accumulator))

		if avg_loss != 0 and self.round == 2:
			self.prev_round_loss = avg_loss
			self.prev_round_time = round(self.global_virtual_clock)

		if self.round >=3:
			logging.info(f"prev_round_info at curr_round: {self.round}, self.prev_round_loss: {self.prev_round_loss}, self.prev_round_time: {self.prev_round_time}")
			diff_loss = self.prev_round_loss - avg_loss
			diff_time = round(self.global_virtual_clock) - self.prev_round_time
			self.time_diff_bet_rounds_min = min(self.time_diff_bet_rounds_min, diff_time)
			self.loss_diff_bet_rounds_max = max(self.loss_diff_bet_rounds_max, diff_loss)
			self.prev_round_loss = avg_loss
			self.prev_round_time = round(self.global_virtual_clock)
			self.loss_diff_bet_rounds.append(diff_loss)
			self.time_diff_bet_rounds.append(diff_time)
			self.prev_ratios.append(self.ratio_prev)

			self.std_dev_loss_diff = statistics.stdev(self.loss_diff_bet_rounds) if len(self.loss_diff_bet_rounds) > 1 else 0
			self.std_dev_time_diff = statistics.stdev(self.time_diff_bet_rounds) if len(self.time_diff_bet_rounds) > 1 else 0
			self.aim_loss_diff_btwn_rounds = self.loss_diff_bet_rounds_max + self.std_dev_loss_diff
			self.aim_time_diff_btwn_rounds = self.time_diff_bet_rounds_min - self.std_dev_time_diff
			logging.info(f"check_lists_round: {self.round}, self.loss_diff_bet_rounds: {self.loss_diff_bet_rounds}, self.time_diff_bet_rounds: {self.time_diff_bet_rounds}, self.std_dev_loss_diff: {self.std_dev_loss_diff}, self.std_dev_time_diff: {self.std_dev_time_diff}, self.aim_loss_diff_btwn_rounds: {self.aim_loss_diff_btwn_rounds}, self.aim_time_diff_btwn_rounds:{self.aim_time_diff_btwn_rounds}")


		logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, Planned participants: " +
					 f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

		# dump round completion information to tensorboard
		if len(self.loss_accumulator):
			self.log_train_result(avg_loss)

		# update select participants
		self.sampled_participants, self.random_participants = self.select_participants(
			select_num_participants=self.args.num_participants, overcommitment=self.args.overcommitment)
		(clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration) = self.tictak_client_tasks(
			self.sampled_participants, self.args.num_participants)

		logging.info(f"Selected participants to run in round {self.round+1}: {clientsToRun}")

		# Issue requests to the resource manager; Tasks ordered by the completion time
		self.resource_manager.register_tasks(clientsToRun)
		self.tasks_round = len(clientsToRun)

		# Update executors and participants
		if self.experiment_mode == commons.SIMULATION_MODE:
			logging.info(f"individual client events: {self.individual_client_events}")
			self.sampled_executors = list(
				self.individual_client_events.keys())
		else:
			self.sampled_executors = [str(c_id)
									  for c_id in self.sampled_participants]

		self.save_last_param()
		self.round_stragglers = round_stragglers
		self.virtual_client_clock = virtual_client_clock
		self.flatten_client_duration = numpy.array(flatten_client_duration)
		self.round_duration = round_duration
		self.model_in_update = 0
		self.test_result_accumulator = []
		self.stats_util_accumulator = []
		self.client_training_results = []
		self.loss_accumulator = []
		self.update_default_task_config()
		
		if self.round >= self.args.rounds:
			self.broadcast_aggregator_events(commons.SHUT_DOWN)
		elif self.round % self.args.eval_interval == 0:
			self.broadcast_aggregator_events(commons.UPDATE_MODEL)
			self.broadcast_aggregator_events(commons.MODEL_TEST)
		else:
			self.broadcast_aggregator_events(commons.UPDATE_MODEL)
			self.broadcast_aggregator_events(commons.START_ROUND)

	def log_train_result(self, avg_loss):
		"""Log training result on TensorBoard
		"""
		self.log_writer.add_scalar('Train/round_to_loss', avg_loss, self.round)
		self.log_writer.add_scalar(
			'FAR/time_to_train_loss (min)', avg_loss, self.global_virtual_clock/60.)
		self.log_writer.add_scalar(
			'FAR/round_duration (min)', self.round_duration/60., self.round)
		self.log_writer.add_histogram(
			'FAR/client_duration (min)', self.flatten_client_duration, self.round)

	def log_test_result(self):
		"""Log testing result on TensorBoard
		"""
		self.log_writer.add_scalar(
			'Test/round_to_loss', self.testing_history['perf'][self.round]['loss'], self.round)
		self.log_writer.add_scalar(
			'Test/round_to_accuracy', self.testing_history['perf'][self.round]['top_1'], self.round)
		self.log_writer.add_scalar('FAR/time_to_test_loss (min)', self.testing_history['perf'][self.round]['loss'],
								   self.global_virtual_clock/60.)
		self.log_writer.add_scalar('FAR/time_to_test_accuracy (min)', self.testing_history['perf'][self.round]['top_1'],
								   self.global_virtual_clock/60.)

	def deserialize_response(self, responses):
		"""Deserialize the response from executor
		
		Args:
			responses (byte stream): Serialized response from executor.

		Returns:
			string, bool, or bytes: The deserialized response object from executor.
		"""
		return pickle.loads(responses)

	def serialize_response(self, responses):
		""" Serialize the response to send to server upon assigned job completion

		Args:
			responses (ServerResponse): Serialized response from server.

		Returns:
			bytes: The serialized response object to server.

		"""
		return pickle.dumps(responses)

	def testing_completion_handler(self, client_id, results):
		"""Each executor will handle a subset of testing dataset
		
		Args:
			client_id (int): The client id.
			results (dictionary): The client test results.

		"""

		results = results['results']

		# List append is thread-safe
		self.test_result_accumulator.append(results)

		# Have collected all testing results

		if len(self.test_result_accumulator) == len(self.executors):
			
			logger.aggregate_test_result(
				self.test_result_accumulator, self.args.task, \
				self.round, self.global_virtual_clock, self.testing_history)
			# Dump the testing result
			with open(os.path.join(logger.logDir, 'testing_perf'), 'wb') as fout:
				pickle.dump(self.testing_history, fout)

			if len(self.loss_accumulator):
				self.log_test_result()

			self.broadcast_events_queue.append(commons.START_ROUND)

	def broadcast_aggregator_events(self, event):
		"""Issue tasks (events) to aggregator worker processes by adding grpc request event
		(e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

		Args:
			event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.
		
		"""
		self.broadcast_events_queue.append(event)

	def dispatch_client_events(self, event, clients=None):
		"""Issue tasks (events) to clients
		
		Args:
			event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.
			clients (list of int): target client ids for event.
		
		"""
		if clients is None:
			clients = self.sampled_executors

		for client_id in clients:
			self.individual_client_events[client_id].append(event)

	def get_client_conf(self, clientId):
		"""Training configurations that will be applied on clients,
		developers can further define personalized client config here.

		Args:
			clientId (int): The client id.

		Returns:
			dictionary: Client training config.

		"""
		conf = {
			'learning_rate': self.args.learning_rate,
		}
		return conf

	def create_client_task(self, executorId):
		"""Issue a new client training task to specific executor
		
		Args:
			executorId (int): Executor Id.
		
		Returns:
			tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

		"""
		next_clientId = self.resource_manager.get_next_task(executorId)
		train_config = None
		# NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
		model = None
		if next_clientId != None:
			config = self.get_client_conf(next_clientId)
			train_config = {'client_id': next_clientId, 'task_config': config}
		return train_config, model

	def get_test_config(self, client_id):
		"""FL model testing on clients, developers can further define personalized client config here.
		
		Args:
			client_id (int): The client id.
		
		Returns:
			dictionary: The testing config for new task.
		
		"""
		return {'client_id': client_id}

	def get_global_model(self):
		"""Get global model that would be used by all FL clients (in default FL)

		Returns:
			PyTorch or TensorFlow module: Based on the executor's machine learning framework, initialize and return the model for training.

		"""
		return self.model

	def get_shutdown_config(self, client_id):
		"""Shutdown config for client, developers can further define personalized client config here.

		Args:
			client_id (int): Client id.
		
		Returns:
			dictionary: Shutdown config for new task.

		"""
		return {'client_id': client_id}

	def add_event_handler(self, client_id, event, meta, data):
		""" Due to the large volume of requests, we will put all events into a queue first.

		Args:
			client_id (int): The client id.
			event (string): grpc event MODEL_TEST or UPLOAD_MODEL.
			meta (dictionary or string): Meta message for grpc communication, could be event.
			data (dictionary): Data transferred in grpc communication, could be model parameters, test result.

		"""
		self.sever_events_queue.append((client_id, event, meta, data))

	def CLIENT_REGISTER(self, request, context):
		"""FL Client register to the aggregator
		
		Args:
			request (RegisterRequest): Registeration request info from executor.

		Returns:
			ServerResponse: Server response to registeration request

		"""

		# NOTE: client_id = executor_id in deployment,
		# while multiple client_id uses the same executor_id (VMs) in simulations
		executor_id = request.executor_id
		executor_info = self.deserialize_response(request.executor_info)
		if executor_id not in self.individual_client_events:
			# logging.info(f"Detect new client: {executor_id}, executor info: {executor_info}")
			self.individual_client_events[executor_id] = collections.deque()
		else:
			logging.info(f"Previous client: {executor_id} resumes connecting")

		# We can customize whether to admit the clients here
		self.executor_info_handler(executor_id, executor_info)
		dummy_data = self.serialize_response(commons.DUMMY_RESPONSE)

		return job_api_pb2.ServerResponse(event=commons.DUMMY_EVENT,
										  meta=dummy_data, data=dummy_data)

	def CLIENT_PING(self, request, context):
		"""Handle client ping requests
		
		Args:
			request (PingRequest): Ping request info from executor.

		Returns:
			ServerResponse: Server response to ping request

		"""
		# NOTE: client_id = executor_id in deployment,
		# while multiple client_id may use the same executor_id (VMs) in simulations
		executor_id, client_id = request.executor_id, request.client_id
		response_data = response_msg = commons.DUMMY_RESPONSE
		
		if len(self.individual_client_events[executor_id]) == 0:
			# send dummy response
			current_event = commons.DUMMY_EVENT
			response_data = response_msg = commons.DUMMY_RESPONSE
		else:
			logging.info(f"====event queue {executor_id}, {self.individual_client_events[executor_id]}")
			current_event = self.individual_client_events[executor_id].popleft() if self.individual_client_events[executor_id] else None
			if current_event == commons.CLIENT_TRAIN:
				response_msg, response_data = self.create_client_task(
					executor_id)
				if response_msg is None:
					current_event = commons.DUMMY_EVENT
					if self.experiment_mode != commons.SIMULATION_MODE:
						self.individual_client_events[executor_id].append(
								commons.CLIENT_TRAIN)
			elif current_event == commons.MODEL_TEST:
				response_msg = self.get_test_config(client_id)
			elif current_event == commons.UPDATE_MODEL:
				response_data = self.get_global_model()
			elif current_event == commons.SHUT_DOWN:
				response_msg = self.get_shutdown_config(executor_id)

		response_msg, response_data = self.serialize_response(
			response_msg), self.serialize_response(response_data)
		# NOTE: in simulation mode, response data is pickle for faster (de)serialization
		response = job_api_pb2.ServerResponse(event=current_event,
										  meta=response_msg, data=response_data)
		if current_event != commons.DUMMY_EVENT:
			logging.info(f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")
		
		return response

	def CLIENT_EXECUTE_COMPLETION(self, request, context):
		"""FL clients complete the execution task.
		
		Args:
			request (CompleteRequest): Complete request info from executor.

		Returns:
			ServerResponse: Server response to job completion request

		"""

		executor_id, client_id, event = request.executor_id, request.client_id, request.event
		execution_status, execution_msg = request.status, request.msg
		meta_result, data_result = request.meta_result, request.data_result

		#print("data_result_of_client: ",data_result) # a pickled result.
		if data_result and (event == commons.CLIENT_TRAIN or event == commons.UPLOAD_MODEL):
			deser_data_result = self.deserialize_response(data_result)
			#print(deser_data_result) # we get the num_fetched_from_cache parameter accurately.
			#print(deser_data_result.keys())
			num_samples = deser_data_result['num_fetched_from_cache']
			self.client_manager.Clients[client_id].update_samples_in_cache(num_samples)

		if event == commons.CLIENT_TRAIN:
			# Training results may be uploaded in CLIENT_EXECUTE_RESULT request later,
			# so we need to specify whether to ask client to do so (in case of straggler/timeout in real FL).
			if execution_status is False:
				logging.error(f"Executor {executor_id} fails to run client {client_id}, due to {execution_msg}")

		# TODO: whether we should schedule tasks when client_ping or client_complete
			if self.resource_manager.has_next_task(executor_id):
				# NOTE: we do not pop the train immediately in simulation mode,
				# since the executor may run multiple clients
				if commons.CLIENT_TRAIN not in self.individual_client_events[executor_id]:
					self.individual_client_events[executor_id].append(
						commons.CLIENT_TRAIN)

		elif event in (commons.MODEL_TEST, commons.UPLOAD_MODEL):
			self.add_event_handler(
				executor_id, event, meta_result, data_result)
		else:
			logging.error(f"Received undefined event {event} from client {client_id}")

		return self.CLIENT_PING(request, context)

	def event_monitor(self):
		"""Activate event handler according to the received new message
		"""
		logging.info("Start monitoring events ...")

		while True:
			# Broadcast events to clients
			if len(self.broadcast_events_queue) > 0:
				current_event = self.broadcast_events_queue.popleft()

				if current_event in (commons.UPDATE_MODEL, commons.MODEL_TEST):
					self.dispatch_client_events(current_event)

				elif current_event == commons.START_ROUND:

					self.dispatch_client_events(commons.CLIENT_TRAIN)

				elif current_event == commons.SHUT_DOWN:
					self.dispatch_client_events(commons.SHUT_DOWN)
					break

			# Handle events queued on the aggregator
			elif len(self.sever_events_queue) > 0:
				client_id, current_event, meta, data = self.sever_events_queue.popleft()
				logging.info("Receive event {} from client {}".format(current_event, client_id))

				if current_event == commons.UPLOAD_MODEL:
					self.client_completion_handler(
						self.deserialize_response(data))
					logging.info("Currently {}/{} clients has finished training".format(len(self.stats_util_accumulator), self.tasks_round))
					if len(self.stats_util_accumulator) == self.tasks_round:
						self.round_completion_handler()

				elif current_event == commons.MODEL_TEST:
					self.testing_completion_handler(
						client_id, self.deserialize_response(data))

				else:
					logging.error(f"Event {current_event} is not defined")

			else:
				# execute every 100 ms
				time.sleep(0.1)

	def stop(self):
		"""Stop the aggregator
		"""
		logging.info(f"Terminating the aggregator ...")
		time.sleep(5)


if __name__ == "__main__":
	aggregator = Aggregator(parser.args)
	aggregator.run()
