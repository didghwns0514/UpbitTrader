########################import py field####################
import pickle
from collections import deque
import os
import typing
import math
import traceback
import random

#########################tensorflow / keras#################
import tensorflow as tf
from keras.models import Sequential #, Model, load_model
from keras.optimizers import Adam
from keras.layers import Dense #, Dropout,  Activation, Flatten, Reshape, Input

from DATA_OPERATION import *
import sub_function_configuration as SUB_F

class Options:
	def __init__(self, env):
		# 인자들 전부 전달해주는 부분
		self.OBSERVATION_DIM = env[0]  # input dimension
		self.H_SIZE = env[1]  # size of hidden layer in bundle
		self.N_SIZE = env[2]  # size of depth of layers
		self.ACTION_DIM = env[3]  # number of actions to take
		"""
		Action dim : 0 - Buy // 1 - Sell // 2 - Hold
		"""

		self.MAX_EPISODE = env[4]  # max number of episodes iteration
		self.GAMMA = env[5]  # discount factor of Q learning
		self.INIT_EPS = env[6]  # initial probability for randomly sampled action
		self.FINAL_EPS = env[7]  # final probability for randomly sampled action
		self.EPS_DECAY = env[8]  # epsilon decay rate
		self.EPS_ANNEAL_STEPS = env[9]  # steps of intervals to decay epsilon
		self.LR = env[10]  # learning rate
		self.MAX_EXPERIENCE = env[11]  # max exprience required for next target Q update
		self.BATCH_SIZE = env[12]  # mini batch size

		self.DROPOUT_KEEP_RATE = env[13]  # keep 할 dropout rate 지정
		self.FRAME_CNT = env[14]
		self.MEMORY_LEN = env[15]

class Trade_agent:
	"""
	https://github.com/llSourcell/Q-Learning-for-Trading
	"""
	# 증권사   /   국가
	TAX_TABLE = {
		'Buy':  1 + ((0.015/100) + 0),
		'Sell' : 1 - ((0.015/100) + (0.3/100)),
		'Hold' : 0
	}

	# while loop fallback counter
	FBK_CNT_MAX = 1000

	MINUTES_WATCH = 60
	ACC_BALANCE = 1
	STOCKS_OWNED = 1
	OBSERVATION_DIM = int(MINUTES_WATCH + ACC_BALANCE + STOCKS_OWNED)

	#                      0,   1,  2,   3,   4,     5,  6,      7,    8,    9,      10,    11,    12,   13,  14,     15
	env = [  OBSERVATION_DIM, 350,  3,   3, 150,  0.95,  1,   1e-5, 0.95,  700, 0.00003,  8000,  1000,  0.5,   3, 100000 ]


	NAME = 'Dqn_model'
	MAX_INT = 400000


	def __init__(self, module=True):

		# @ load model configuration
		self.option = Options(self.env)

		# @ set keras environment
		self.bool_train = None
		if module == True:
			self.option.DROPOUT_KEEP_RATE = 1
			self.config = tf.ConfigProto(
				device_count={'GPU': 0})
			self.eps = self.option.FINAL_EPS
			self.bool_train = False

		else:
			self.config = tf.ConfigProto(
				device_count={'GPU': 1})
			self.eps = self.option.INIT_EPS

			# @ train 여부
			ans = input('train y/n')
			self.bool_train = True if ans == 'y' else False
			if self.bool_train:
				if input('do you want to set eps to 0? y/n') == 'y':
					self.eps = self.option.FINAL_EPS
				else:
					pass


		# @ main dir for saving
		self.dir = str(os.getcwd() + '\\DQN_model__checkpoint').replace('/', '\\')
		self.file = self.dir + '\\' + 'DQN_model.h5'
		if os.path.isdir(self.dir):
			pass
		else:
			os.mkdir(self.dir)

		# @ separate graph from initial
		self.MAIN_GRAPH = tf.Graph()
		with self.MAIN_GRAPH.as_default() as g:
			self.MAIN_SESS = tf.Session(graph=g, config=self.config)


		# @ weights
		self.MAIN_MODEL = None
		self.TARGET_MODEL = None
		self.FUNC_load_model()

		# @ local variables to use
		self.memory = deque(maxlen=self.option.MEMORY_LEN)
		self.observation = []
		self.reward = None
		self.action = None
		self.global_cnt = 0
		self.target_f = None
		#-------------------------------------
		self.start_price = None # start price of the obeservation dimentsion, [0] index // to normalize
		self.end_price = None
		self.acc_balance = None # balance in the account
		self.CONST_acc_balance = None # balance started with
		self.stocks_own = None # stocks owned "now"
		self.stocks_ownList = [] # stocks owning price list
		# ------------------------------------


	def FUNC_set_init_values(self, acc_balance:float, stocks_ownList:int): # setter
		"""

		:param acc_balance: acc balance owned
		:param stocks_ownList: list of stocks owned
		:return:
		"""

		self.acc_balance = acc_balance # balance in the account
		self.CONST_acc_balance = acc_balance

		self.stocks_ownList = stocks_ownList # stocks owned "now"


	def FUNC_update_target(self):
		"""
		update target graph with local graph of Q
		:return:
		"""
		self.TARGET_MODEL.set_weights(self.MAIN_MODEL.get_weights())


	def FUNC_memory_remember(self, state, action, reward, next_state):
		"""
		append memory to save... and train
		:param state: observed environment "now"
		:param action: action chosen at the "now" environment
		:param reward: reward given as the result of the action chosen
		:param next_state: next state when action is taken
		:return:
		"""
		self.memory.append((state, action, reward, next_state))


	def FUNC_model_action(self, state, _random=True):
		"""

		:param state: observation of "now"
		:param _random: either randomized result is needed, default if True
		:return: argmax of action / all the actions
		"""
		if np.random.rand() <= self.eps and _random:
			return random.randrange(0, self.option.ACTION_DIM), None

		with self.MAIN_GRAPH.as_default() as g:
			with self.MAIN_SESS.as_default() as sess:
				inputz = np.array(state)
				inputz = inputz.reshape(-1, int(self.option.OBSERVATION_DIM))

				act_values = self.MAIN_MODEL.predict(inputz)

				return np.argmax(act_values[0]), act_values  # returns actions


	def FUN_model_replay(self):
		"""
		vectorized implementation; 30x speed up compared with for loop

		https://github.com/jinyeong/study_RL-keras/blob/master/2-cartpole/1-dqn/cartpole_dqn.py
		https://wonseokjung.github.io/2018-03-23-RLCODE-Breakoutdqn_keras/
		http://solarisailab.com/archives/486?ckattempt=1

		# 알파 제로 만들기
		https://keraskorea.github.io/posts/2018-10-23-%ED%8C%8C%EC%9D%B4%EC%8D%AC%EA%B3%BC_%EC%BC%80%EB%9D%BC%EC%8A%A4%EB%A5%BC_%EC%9D%B4%EC%9A%A9%ED%95%9C_%EC%95%8C%ED%8C%8C%EC%A0%9C%EB%A1%9C_%EB%A7%8C%EB%93%A4%EA%B8%B0/
		"""
		"""
		where actual training and updating target Q happens!
		"""

		with self.MAIN_GRAPH.as_default() as g:
			with self.MAIN_SESS.as_default() as sess:

				if self.eps > self.option.FINAL_EPS and self.global_cnt % self.option.EPS_ANNEAL_STEPS == 0:
					"""
					decay eps and moving to greedy selection
					"""
					self.eps *= self.option.EPS_DECAY

				if self.global_cnt % self.option.FRAME_CNT == 0 and \
						len(self.memory) > self.option.BATCH_SIZE:

					mini_batch = random.sample(self.memory, self.option.BATCH_SIZE)

					# @ get samples
					states = np.zeros((self.option.BATCH_SIZE,
									   self.option.OBSERVATION_DIM))
					next_states = np.zeros((self.option.BATCH_SIZE,
											self.option.OBSERVATION_DIM))
					actions, rewards = [], []
					for i in range(self.option.BATCH_SIZE):
						states[i] = mini_batch[i][0]
						actions.append(mini_batch[i][1])
						rewards.append(mini_batch[i][2])
						next_states[i] = mini_batch[i][3]

					# @ local network
					local_val = self.MAIN_MODEL.predict(states)

					# @ target network
					target_val = self.TARGET_MODEL.predict(next_states)

					# @ current local network update values / from Bellman equ
					for i in range(self.option.BATCH_SIZE):
						local_val[i][actions[i]] = rewards[i] + \
												   (self.option.GAMMA * np.amax(target_val[i]))

					self.MAIN_MODEL.fit(states,
										local_val,
										batch_size=self.option.BATCH_SIZE,
										epochs=1, verbose=0)  # x: input value, y:target value

					if (self.global_cnt % self.option.MAX_EXPERIENCE == 0):
						self.FUNC_update_target()

					if self.global_cnt >= Trade_agent.MAX_INT:
						# reset just in case of integer overflow..?
						self.global_cnt = 0


	def FUNC_load_model(self):
		# https://3months.tistory.com/150
		with self.MAIN_GRAPH.as_default() as g:
			with self.MAIN_SESS.as_default() as sess:
				self.MAIN_MODEL = self.FUNC_build_model(n_obs=self.option.OBSERVATION_DIM,
														n_action=self.option.ACTION_DIM,
														n_hidden_layer=self.option.N_SIZE,
														n_neuron_per_layer=self.option.H_SIZE)
				self.TARGET_MODEL = self.FUNC_build_model(n_obs=self.option.OBSERVATION_DIM,
														  n_action=self.option.ACTION_DIM,
														  n_hidden_layer=self.option.N_SIZE,
														  n_neuron_per_layer=self.option.H_SIZE)

				if os.path.isfile(self.file):
					self.MAIN_MODEL.load_weights(self.file)
					print(f'loaded self.MAIN_MODEL.summary() : {self.MAIN_MODEL.summary()}')

				else:
					print(f'built self.MAIN_MODEL.summary() : {self.MAIN_MODEL.summary()}')

				self.FUNC_update_target()


	def FUNC_save_model(self):
		# https://3months.tistory.com/150
		with self.MAIN_GRAPH.as_default() as g:
			with self.MAIN_SESS.as_default() as sess:
				self.MAIN_MODEL.save_weights(self.file)


	def FUNC_build_model(self, n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32, activation='relu', loss='mse'):
		""" A multi-layer perceptron """
		model = Sequential()
		model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation, name=self.NAME + 'Dense_input'))
		for i in range(n_hidden_layer):
			model.add(Dense(n_neuron_per_layer, activation=activation, name=self.NAME + \
																			'_' + 'Dense' + '_' + str(i)))
		model.add(Dense(n_action, activation='linear', name=self.NAME + 'Dense_final'))
		model.compile(loss=loss, optimizer=Adam())

		return model


	def FUNC_predict_model(self, state:list):
		"""
		actual 'prediction only' function, returns action
		:param state: state tobe predicted multiple times.. use FUNC_model_action function defined in the class
					  start value must be located in the first index to latest at the end...
		:return: Action - ----always calculate as scope of "NOW" time, if to buy more or not...
						  1) action chosen
						  2) numbers taken
		"""
		"""
		[used func / variable]
		self.start_price = None # start price of the obeservation dimentsion, [0] index // to normalize
		self.acc_balance = None # balance in the account
		self.stocks_own = None # stocks owned at the current
		FUNC_set_init_values
		"""

		assert len(state) == int(Trade_agent.MINUTES_WATCH)

		# @ variables used
		self.start_price = state[0]
		self.end_price = state[-1]

		self.action_cnt = 1
		tmp_decod_act = None # (n-1) action

		fallback_cnt = 0 # for inf-while loop fallback

		while True:

			if fallback_cnt >= Trade_agent.FBK_CNT_MAX:
				break

			# get altered features
			tmp_data = self.FUNC_concat_observe(state_stocks=state,
												start_price=self.start_price,
												acc_balance=self.acc_balance,
												stocks_own=len(self.stocks_ownList))

			# @ decoded action
			num_action, action_arr = self.FUNC_model_action(state=tmp_data,
															_random=self.bool_train)
			tmp_sub_decod_act = self.FUNC_decode_action(num_action) # (n) action


			# @ up global counter
			self.global_cnt += 1

			# @ calc iter, next state, reward
			bool_nextIter, nextState, reward_calculated = \
				            self.FUNC_update_state(decoded_action=tmp_sub_decod_act,
												   stock_state=state)

			# @ run memory remember
			if self.bool_train and self.global_cnt % self.option.FRAME_CNT == 0:
				self.FUNC_memory_remember(state=tmp_data,
										  action=num_action,
										  reward=reward_calculated,
										  next_state=nextState)
			# @ first variable assign check
			if tmp_decod_act == None:
				tmp_decod_act = copy.deepcopy(tmp_sub_decod_act)

			# @ if different action chosen
			if tmp_decod_act != tmp_sub_decod_act:
				break
			else:
				# @ update n-1
				tmp_decod_act = copy.deepcopy(tmp_sub_decod_act)

				# @ update  by setter
				# bool_nextIter, _ = self.FUNC_update_state(bool_update=True,
				# 										  decoded_action=tmp_sub_decod_act)
				if not bool_nextIter: # not allowed action
					break

			# @ rise fallback cnt
			fallback_cnt += 1
			# @ assertions after each loops
			assert self.acc_balance >= 0
			assert len(self.stocks_ownList) >= 0


		return tmp_decod_act, self.action_cnt



	def FUNC_update_state(self, decoded_action:str, stock_state:list )\
			->typing.Tuple[bool or None, list or np.array, float or None]:
		"""
		update state based on bool_update(to allow action to update state if it's possible)
		:param decoded_action: decoded action
		:param stock_state: variable of original stocks minute price
		:return: [action allowed, (None / next state), reward calculated]
		"""

		assert stock_state != None and isinstance(stock_state, list)

		# @ local variables
		tmp_next_data = None
		tmp_iterNext = True
		tmp_rwd_calc = None

		tmp_acc_balance = copy.deepcopy(self.acc_balance)
		tmp_stocks_own = copy.deepcopy(len(self.stocks_ownList))


		if decoded_action == 'Hold':
			pass
		elif decoded_action == 'Buy':
			tmp_acc_balance -= Trade_agent.TAX_TABLE['Buy'] * self.end_price
			tmp_stocks_own += 1
		elif decoded_action == 'Sell':
			tmp_acc_balance += Trade_agent.TAX_TABLE['Sell'] * self.end_price
			tmp_stocks_own -= 1

		tmp_next_data = self.FUNC_concat_observe(state_stocks=stock_state,
											start_price=self.start_price,
											acc_balance=tmp_acc_balance,
											stocks_own=tmp_stocks_own)


		if decoded_action == 'Hold':
			#return False, None, self.FUNC_reward(bool_allowed=True) # allowed exception for Hold
			tmp_iterNext = False
			tmp_rwd_calc = self.FUNC_reward(bool_allowed=True)

		elif decoded_action == 'Buy':
			calc_price = Trade_agent.TAX_TABLE['Buy'] * self.end_price
			if self.acc_balance >= calc_price:
				self.acc_balance -= calc_price
				self.FUNC_inventory = self.end_price # setter
				self.action_cnt += 1
				tmp_iterNext = True
				tmp_rwd_calc = self.FUNC_reward(bool_allowed=True)
			else:
				tmp_iterNext = False
				tmp_rwd_calc = self.FUNC_reward(bool_allowed=False)


		elif decoded_action == 'Sell':
			calc_price = Trade_agent.TAX_TABLE['Sell'] * self.end_price
			if self.stocks_own < 1:
				tmp_iterNext = False
				tmp_rwd_calc = self.FUNC_reward(bool_allowed=False)
			else:
				self.acc_balance += calc_price
				_ = self.FUNC_inventory # getter // pop
				self.action_cnt += 1
				tmp_iterNext = True
				tmp_rwd_calc = self.FUNC_reward(bool_allowed=True)

		return tmp_iterNext, tmp_next_data, tmp_rwd_calc

	@property
	def FUNC_inventory(self): # getter

		return self.stocks_ownList.pop(0)

	@FUNC_inventory.setter
	def FUNC_inventory(self, new_price): # setter

		tmp_ownList_cpy = copy.deepcopy(self.stocks_ownList)
		tmp_ownList_cpy.append(new_price)
		mean_val = sum(tmp_ownList_cpy) / len(tmp_ownList_cpy)

		self.stocks_ownList = [ mean_val for _ in tmp_ownList_cpy]


	def FUNC_reward(self, bool_allowed:bool, RWD_TYPE:int=1) ->float:
		"""

		:param bool_allowed:
		:return: reward after using decoded action as the input
				 -> if wrong action /and right action, the rewards are changed
		"""
		# @ local values
		reward = 0

		# @ if logic by allowed action
		if bool_allowed: # allowed action

			if RWD_TYPE == 0: # balance only
				reward = math.log(self.acc_balance) - math.log(self.CONST_acc_balance)

			elif RWD_TYPE == 1:
				calc_spend_price = 0 if not self.stocks_ownList else sum(self.stocks_ownList)
				calc_now_price = 0 if not self.stocks_ownList else len(self.stocks_ownList) * self.end_price
				reward = math.log(calc_now_price) - math.log(calc_spend_price)

		else: # not allowed action
			reward = -1

		return reward



	def FUNC_concat_observe(self, state_stocks:list, start_price:int, acc_balance:float,  stocks_own:int) ->np.ndarray:
		"""

		:param state_stocks:
		:param start_price: start price of state_stocks
		:param acc_balance:
		:param stocks_own:
		:return: concatnate data for DQN input
		"""
		tmp_list = []
		tmp_list.extend(state_stocks)
		tmp_list.extend([float(acc_balance / start_price)]) # calculate relative balance against start stock price
		tmp_list.extend([int(stocks_own)])

		tmp_arr = np.array(tmp_list)
		tmp_arr = tmp_arr.reshape(-1, int(self.option.OBSERVATION_DIM))

		return tmp_arr


	def FUNC_decode_action(self, action_chosen:int) ->str:
		"""

		:param action_chosen: action index
		:return: string of action decoded
		"""
		"""
		Action dim : 0 - Buy // 1 - Sell // 2 - Hold
		"""
		if action_chosen == 0:
			return 'Buy'
		elif action_chosen == 1:
			return 'Sell'
		elif action_chosen == 2:
			return 'Hold'


def Session():

	# # @ import for db
	import sqlite3

	# @ stock prediction wrapper class
	prediction_agent = Trade_agent(module=True)

	## current working python directory
	current_wd = os.getcwd().replace('/', '\\')

	## directory tobe used
	dir_db__folder = current_wd + '\\' + 'PREDICTER__DATABASE_single'
	dir_article__folder = current_wd + '\\' + 'PREDICTER__ARTICLE_check'

	## db and pickle
	dir_db__file = dir_db__folder + '\\' + 'SINGLE_DB.db'
	dir_pickle__file = dir_db__folder + '\\' + 'parsed_list_pickle.p'
	dir_article__file = dir_db__folder + '\\' + 'pickle.p'
	dir_pickle_skipped__file = dir_db__folder + '\\' + 'stock_list_pickle__skipped.p'

	# @ Sqlite object
	sqlite_closure = sqlite_capture(dir_db__file)
	list_of_codes = sqlite_closure(get_codes=True)

	#####################
	## var for loading!
	pickle_article = None
	with open(dir_article__file, 'rb') as file:
		pickle_article = copy.deepcopy(pickle.load(file))

	pickle_visited_list = None
	with open(dir_pickle_skipped__file, 'rb') as file:
		pickle_visited_list = copy.deepcopy(pickle.load(file))
	#####################

	df__kospi = sqlite_closure(_stock_code=str(226490))
	df__dollar = sqlite_closure(_stock_code=str(261250))
	MUST_WATCH_LIST = ["226490", "261250"]  # "252670"
	"""                KODEX 코스피, KODEX 미국달러선물 레버리지, KODEX 200선물 인버스 2X"""

	for stock_code in list_of_codes:
		pushLog(dst_folder='SESSION__PREDICTER__ML_MAIN',
				exception=True,
				memo=f'entered stock : {str(stock_code)}')

		# @ skip must watch list
		if stock_code in MUST_WATCH_LIST:
			pushLog(dst_folder='SESSION__PREDICTER__ML_MAIN',
					exception=True,
					memo=f'stock code in MUST_WATCH_LIST')
			continue

		# @ check empty dataframe
		main_Stk_df = sqlite_closure(_stock_code=str(stock_code))
		if main_Stk_df.empty:
			pushLog(dst_folder='SESSION__PREDICTER__ML_MAIN',
					exception=True,
					memo=f'stock code did not match standards, returned None type')
			continue

		mainStk_dt_start__obj = main_Stk_df.date.min() + datetime.timedelta(days=10)
		mainStk_dt_end__obj = main_Stk_df.date.max() - datetime.timedelta(days=1)

		#################################
		# If all has passed : start from here!
		#################################
		while mainStk_dt_start__obj <= mainStk_dt_end__obj:
			tmp_dt_start__obj = SUB_F.FUNC_dtRect(mainStk_dt_start__obj, "9:00")
			tmp_dt_end__obj = SUB_F.FUNC_dtRect(mainStk_dt_start__obj, "15:30")

			# checking break in the middle
			tmp_break_bool = False

			print('\n' * 2)
			print(f'=#' * 20)
			print(f'tmp_dt_start__obj : {tmp_dt_start__obj}')
			print(f'tmp_dt_end__obj : {tmp_dt_end__obj}')

			f_kospi = sweep_day(df=df__kospi,
								stock_code=str(226490),
								start_date=tmp_dt_start__obj,
								end_date=tmp_dt_end__obj,
								_type='data',
								)

			f_dollar = sweep_day(df=df__dollar,
								 stock_code=str(261250),
								 start_date=tmp_dt_start__obj,
								 end_date=tmp_dt_end__obj,
								 _type='data',
								 )

			# f_ans = sweep_day(df=main_Stk_df,
			# 			stock_code=stock_code,
			# 			start_date=tmp_dt_start__obj,
			# 			end_date=tmp_dt_end__obj,
			# 			_type='answer',
			# 			)

			f_data = sweep_day(df=main_Stk_df,
							   stock_code=stock_code,
							   start_date=tmp_dt_start__obj,
							   end_date=tmp_dt_end__obj,
							   _type='data',
							   )

			while True:
				try:
					hash_kospi, t1 = f_kospi.__next__()
					hash_dollar, t2 = f_dollar.__next__()
					hash_data, t3 = f_data.__next__()
					# hash_ans, t4 = f_ans.__next__()

					if len(list(set([t1, t2, t3]))) != 1:
						pushLog(dst_folder='PREDICTER__ML_MAIN',
								lv='ERROR',
								module='Session',
								exception=True,
								memo=f'time stamp different by generators')
						tmp_break_bool = True
						print(f'datestamp list isnt in sink...!')
						break

					elif list(filter(lambda x: x == None, [hash_kospi, hash_dollar, hash_data])):
						tmp_break_bool = True
						print(f'value returned from generator is corrupt...!')
						break

					if SQ_check_opDay(t1):
						print(f'weekday, proceeding...!')
					else:
						print(f'weekend, skipping...!')
						tmp_break_bool = True
						break
					rtn = prediction_agent._stock_op_wrapper(stock_code=stock_code,
															 hash_stock=hash_data,
															 hash_kospi=hash_kospi,
															 hash_dollar=hash_dollar,
															 _today=t1,
															 hash_article=pickle_article)

					log, data = rtn  ## data 분석 proceed -> plot graph
					rtn_checker = ReturnWrap._type(_type='PREDICTER_TEST', rtn_val=rtn)
					if rtn_checker:
						print(f'predictable! -> log message : {log}')

					else:  # unpredictable
						print(f'un-predictable! -> log message : {log}')
						tmp_break_bool = True
						break

					## add success log
					################################################
					pushLog(dst_folder='PREDICTER__ML_MAIN',
							lv='INFO',
							module='Session',
							exception=True,
							memo=f'date used : {str(t1)} \
								  \n- normally passed')

				except StopIteration as se:
					print(f'error in sweeping singe day : {se}')
					traceback.print_exc()
					pushLog(dst_folder='PREDICTER__ML_MAIN',
							lv='ERROR',
							module='Session',
							exception=True,
							exception_msg=str(se),
							memo=f'StopIteration exception')
					# input(f'type any to continue to next!')
					# tmp_break_bool = True
					break

			if tmp_break_bool == False:

				# @ get prediction result
				check_stock_code = prediction_agent.nestgraph.NG__check_stkcode(stock_code)
				if check_stock_code:
					print(f'stock code exists in nested graph')

					tmp_pred_datetime_dict = \
						prediction_agent.nestgraph.NG__get_prediction_dict(stock_code)

					tmp_model_status = prediction_agent.nestgraph.NG__get_accuracy(stock_code)

					if tmp_pred_datetime_dict:
						# @ plot graph and save
						############################
						# Plotting only done in the session
						#
						############################
						tmp_single_day_df = main_Stk_df.loc[(main_Stk_df.date >= tmp_dt_start__obj) & (
								main_Stk_df.date <= tmp_dt_end__obj)]
						print(f'start plotting graph')
						FUNC__save_image(start_day_str=tmp_dt_start__obj,
										 model_status=tmp_model_status,
										 dataframe=tmp_single_day_df,
										 pred_dict=tmp_pred_datetime_dict,
										 stock_code=stock_code)
						pass

					else:
						print(f'iter to next day - no predictions available')
				else:
					print(f'no stock code available in nested graph')

			# add day
			mainStk_dt_start__obj += datetime.timedelta(days=1)

	print(f'total execution finished!')


def FUNC__save_image(start_day_str, model_status, dataframe, pred_dict, stock_code):
	"""

	:param start_day_str: predction training datetime
	:param model_status: model status container, follow hierarchy call tree to check
						 -> [ [ model state, model accuracy  ], [ , ] ..... [ , ] ]
	:param dataframe:
	:param pred_dict:
	:param stock_code:
	:return:
	"""

	import matplotlib.pyplot as plt
	import random

	folder_location = (os.getcwd() + '\\PREDICTER__Image_result').replace('/', '\\')
	if not os.path.isdir(folder_location):
		os.mkdir(folder_location)
	file_location = folder_location + '\\' + str(stock_code) + '_' + str(
		start_day_str.strftime("%Y-%m-%d")) + '.png'

	# @ picture
	fig = plt.figure(figsize=(100, 50))
	ax1 = fig.add_subplot(111)

	# @ set title
	tmp_title = ''
	for stat, acc in model_status:
		tmp_title += 'state : ' + str(stat)
		tmp_title += '\n' + 'acc : ' + str(acc)
	plt.title(tmp_title)

	# @ add original dateimte - value plot
	# ax1 = dataframe.plot(x='date', y='open', figsize = (100, 50), grid=True, Linewidth=1, fontsize=5)
	dataframe.plot(x='date', y='open', figsize=(80, 50), grid=True, Linewidth=1, fontsize=5, ax=ax1)

	# https://datascienceschool.net/view-notebook/372443a5d90a46429c6459bba8b4342c/
	# @ plot rand
	tmp_plot_item = ['b', 'r', 'g', 'k', 'c', 'm']

	for date_key in pred_dict:
		dot_color = random.choice(tmp_plot_item)
		tmp_x = [date_key + datetime.timedelta(minutes=i + 1) for i in range(0, len(pred_dict[date_key]))]
		tmp_y = pred_dict[date_key]

		ax1.plot_date(tmp_x, tmp_y,
					  color=dot_color,
					  marker='o',
					  linestyle='solid',
					  alpha=0.5)  # marker='None', marker='o'
		ax1.axvline(x=date_key,
					color='r',
					linestyle='--',
					linewidth=1,
					alpha=0.3)
		ax1.axvline(x=date_key + datetime.timedelta(minutes=len(tmp_y)),
					color='b',
					linestyle='--',
					linewidth=1,
					alpha=0.3)

	try:
		fig.savefig(file_location, dpi=150)
		plt.close(fig)

		print(f'plotting successfully saved!')
	except Exception as e:
		import traceback
		print(f'failed to save the plotting...: {e}')
		traceback.print_exc()

	# @ try deleting them
	try:
		fig = None
		del fig

		print(f'successful deleting them in FUNC__save_image')
	except Exception as e:
		print(f'error in  deleting them in FUNC__save_image... {e}')


if __name__ == '__main__':
	# training begin
	Session()
