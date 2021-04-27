import sqlite3
import numpy as np
import pandas as pd
from sub_function_configuration import *


def SQ__parse_answer(ori_df_whole, dt_now__obj, minute_forward, stock_code):

	"""

	:param ori_df_whole: original whole data
	:param dt_now__obj: datetime that is "now" at the current!
	:param minute_forward: minute forward to parse data
	:return: dictionary [time : value]
	"""

	return_dict = {}

	## no need to skip no next week for the future data
	if dt_now__obj + datetime.timedelta(minutes=minute_forward) <= FUNC_dtRect(dt_now__obj, "15:30"):
		df_target = ori_df_whole.loc[(ori_df_whole.date >= dt_now__obj) & (
				ori_df_whole.date < dt_now__obj + datetime.timedelta(minutes=minute_forward))]

		dict_target = SQ_fill_missing(ori_dict=SQ__dfToDict(df_target),
									  start_time_obj=dt_now__obj,
									  end_time_obj=dt_now__obj + datetime.timedelta(minutes=minute_forward))
		return_dict.update(dict_target)

	## need to obtain next week data
	else:
		tmp_fwd_timeshift = FUNC_dtRect(dt_now__obj,"15:30") - dt_now__obj
		tmp_fwd_deltaMinutes = divmod(tmp_fwd_timeshift.total_seconds(), 60)[0]
		tmp_calc_minutes = minute_forward - tmp_fwd_deltaMinutes # amount of next week's minute data
		datetime_target = None
		if dt_now__obj.weekday() == 4:  # 금요일
			datetime_target = dt_now__obj + datetime.timedelta(days=3)

		else:
			datetime_target = dt_now__obj + datetime.timedelta(days=1)

		df_trg_stTime = FUNC_dtRect(datetime_target,"9:00")
		df_trg_endTime = df_trg_stTime + datetime.timedelta(minutes=tmp_calc_minutes)
		df_target = ori_df_whole.loc[
			(ori_df_whole.date >= df_trg_stTime) & (ori_df_whole.date <= df_trg_endTime)]

		df_now = ori_df_whole.loc[(ori_df_whole.date >= dt_now__obj) & (
				ori_df_whole.date < FUNC_dtRect(dt_now__obj,"15:30"))]

		# dictionary
		dict_df_now = SQ_fill_missing(SQ__dfToDict(df_now), dt_now__obj,
													  FUNC_dtRect(dt_now__obj, "15:30"))
		dict_df_target = SQ_fill_missing(SQ__dfToDict(df_target),
														 df_trg_stTime, df_trg_endTime)

		# @ update
		return_dict.update(dict_df_now)
		return_dict.update(dict_df_target)

	assert len(list(return_dict.keys())) == int(minute_forward)

	return return_dict


def SQ__parse_sqData(ori_df_whole, dt_now__obj, hours_duration_back, stock_code):

	"""

	:param ori_df_whole: original whole dataframe to parse
	:param dt_now__obj: datetime that is "now" at the current!
	:param hours_duration_back: hours back to get the data
	:return: dictionary of stock data,  /// time : { price, volume }
	         -> if all df are missing, return None
	"""
	return_dict = {}

	dtList_toParse = FUNC_datetime_backward(dt_now__obj, hours_duration_back)

	tmp_missingDates = dict()

	#print(f'stock_number : {stock_code}')
	#print(f'dtList_toParse : {dtList_toParse}')

	for i in range(len(dtList_toParse)):

		# @ declare local values inside forloop
		tmp_df = None
		tmp_missingDates[i] = {

			'missing':False,
			'missing_list' : None,
			'data':{}

		}

		#print(f'\n'*2)
		#print(f'i th iteration : {i}')
		#print(f'parse from {dtList_toParse[i][0]} to {dtList_toParse[i][1]}')

		# if i == 0: # when i == 0, empty dataframe is a problem
		# 	tmp_df = ori_df_whole.loc[(ori_df_whole.date >= dtList_toParse[i][0]) & (
		# 				ori_df_whole.date <= dtList_toParse[i][1])]

		tmp_df = ori_df_whole.loc[(ori_df_whole.date >= dtList_toParse[i][0]) & (
						ori_df_whole.date <= dtList_toParse[i][1])]

		# print(f'length of dataframe parsed : {len(tmp_df)}')

		if tmp_df.empty:
			#print(f'it is empty')
			tmp_nums_iter = int((dtList_toParse[i][1] - dtList_toParse[i][0]).total_seconds() // 60) + 1
			tmp_list = [ dtList_toParse[i][0] + datetime.timedelta(minutes=k)  \
						     for k in range(0, tmp_nums_iter)]
			# print(f'tmp_nums_iter : {tmp_nums_iter}')
			# print(f'tmp_list : {tmp_list}')

			# @ record missing
			tmp_missingDates[i]['missing'] = True
			tmp_missingDates[i]['missing_list'] = tmp_list


		else:
			#print(f'it is not empty')
			tmp_dict = SQ_fill_missing(SQ__dfToDict(tmp_df),
													   dtList_toParse[i][0],
													   dtList_toParse[i][1])
			return_dict.update(tmp_dict)

			# @ record missing
			tmp_missingDates[i]['data'] = copy.deepcopy(tmp_dict)


	# @ use function
	if tmp_missingDates: # value exists in container
		rtn_after_fill = SQ__fill_missing_df(missing_dict=tmp_missingDates,
											   temp_dict=return_dict)
		if rtn_after_fill == None : # all are empty df
			return None

		return_dict.update(rtn_after_fill)

	return return_dict


def SQ__fill_missing_df(missing_dict:dict, temp_dict:dict):
	"""

	:param missing_dict: missing record of dictionary
	                             - consists of index key including data -> go to : SQ__parse_sqData
									tmp_missingDates[i] = {

									'missing':False,
									'missing_list' : None,
									'data':None

									 }
	:param temp_dict: temporary filled dictionary
	:return: Action - fill missing data from df when its empty initially,
	                  compare against temporary filled dataframe -> dict result
	                  -> if all df are missing, return None
	"""
	# @ dict
	return_dict = copy.deepcopy(temp_dict)

	# @ sort index key
	index_list = sorted(list( missing_dict.keys() ))

	# @ check if every values are missing (False)
	tmp_bool_check = all([ val['missing'] == True   \
					   for key, val in zip(missing_dict.keys(), missing_dict.values())] )
	if tmp_bool_check:
		return None

	# @ calc missing list
	for index_key in index_list:
		"""
		first index missing -> get from... next index
		not recommended however, it is necessary
		"""
		if index_key == 0 and  missing_dict[index_key]['missing']:

			filter_index = [ ind for ind in index_list \
								   if ind > index_key  if missing_dict[ind]['missing'] == False ]
			assert filter_index # has more than one item
			nonMissing_key = min(filter_index)

			# get farback key from next index
			date_key__farback  = sorted(list(missing_dict[int(nonMissing_key)]['data'].keys()))[0]
			value_selected = missing_dict[int(nonMissing_key)]['data'][date_key__farback]
			value_selected['volume'] = 0 # reset only the volume, keep price factor

			# returned result
			tmp_rtn_dict = {key : value_selected for key in missing_dict[index_key]['missing_list']}

			# update return dictionary
			return_dict.update(tmp_rtn_dict)

			# update missing record dictionary
			missing_dict[index_key]['missing'] = False
			missing_dict[index_key]['data'].update(tmp_rtn_dict)

		elif index_key != 0 and missing_dict[index_key]['missing']:
			filter_index = [ ind for ind in index_list \
								   if ind < index_key  if missing_dict[ind]['missing'] == False ]
			assert filter_index # has more than one item
			nonMissing_key = max(filter_index)

			# get latest key from before index
			date_key__latest = sorted(list(missing_dict[int(nonMissing_key)]['data'].keys()))[-1]
			value_selected = missing_dict[int(nonMissing_key)]['data'][date_key__latest]
			value_selected['volume'] = 0  # reset only the volume, keep price factor

			# returned result
			tmp_rtn_dict = {key : value_selected for key in missing_dict[index_key]['missing_list']}

			# update return dictionary
			return_dict.update(tmp_rtn_dict)

			# update missing record dictionary
			missing_dict[index_key]['missing'] = False
			missing_dict[index_key]['data'].update(tmp_rtn_dict)

	return return_dict



def SQ__dfToDict(dataframe):

	tmp_dictionary_return = {}

	for row_tuple in dataframe.itertuples():
		# tmp_dictionary_return[row_tuple.date.strftime('%Y%m%d%H%M%S')] = {'price': row_tuple.open,
		# 																  'volume': row_tuple.volume}
		tmp_dictionary_return[row_tuple.date] = {'price': row_tuple.open,
																		  'volume': row_tuple.volume}

	return tmp_dictionary_return


def SQ_fill_missing(ori_dict, start_time_obj, end_time_obj):
	####여기서 missing 나온다
	# try:
	return_dict = copy.deepcopy(ori_dict)

	tmp_list_missing_dt = []

	tmp_list_exist_dt = list(ori_dict.keys())
	tmp_list_exist_dt.sort()

	# tmp_dt_srt_stamp = tmp_list_exist_dt[0]  # 첫 데이터
	# tmp_dt_end_stamp = tmp_list_exist_dt[-1]  # 마지막 데이터
	# tmp_dt_srt_stamp_obj = FUNC_dtRect(FUNC_dtSwtich(tmp_dt_srt_stamp))
	# tmp_dt_end_stamp_obj = FUNC_dtRect(FUNC_dtSwtich(tmp_dt_end_stamp))

	tmp_dt_srt_stamp = tmp_list_exist_dt[0]  # 첫 데이터
	tmp_dt_end_stamp = tmp_list_exist_dt[-1]  # 마지막 데이터
	tmp_dt_srt_stamp_obj =  FUNC_dtRect(tmp_dt_srt_stamp)  # 첫 데이터
	tmp_dt_end_stamp_obj = FUNC_dtRect(tmp_dt_end_stamp)  # 마지막 데이터

	if tmp_dt_srt_stamp_obj <= tmp_dt_end_stamp_obj:
		before_price = None
		before_volume = None
		while tmp_dt_srt_stamp_obj <= tmp_dt_end_stamp_obj:  # datetime obj끼리 비교 while 문이라 위험??
			# @ 처음은 list에서 뽑아왔으므로 있다
			#tmp_dt_srt__key = FUNC_dtSwtich(tmp_dt_srt_stamp_obj)
			if tmp_dt_srt_stamp_obj in ori_dict:
				before_price = ori_dict[tmp_dt_srt_stamp_obj]['price']
				before_volume = ori_dict[tmp_dt_srt_stamp_obj]['volume']
			else:
				tmp_list_missing_dt.append(tmp_dt_srt_stamp_obj)
				return_dict[tmp_dt_srt_stamp_obj] = {'price': before_price, 'volume': 0}  # 'volume': before_volume

			tmp_dt_srt_stamp_obj += datetime.timedelta(minutes=1)

	# 1) 뒤쪽에서 값이 missing된 경우
	tmp_dt_end_stamp_obj = FUNC_dtRect(tmp_dt_end_stamp)
	tmp_end_stub_price = ori_dict[tmp_dt_end_stamp_obj]['price']
	tmp_end_stub_volume = ori_dict[tmp_dt_end_stamp_obj]['volume']

	while tmp_dt_end_stamp_obj <= end_time_obj:
		#tmp_dt_end_stamp_obj_convert = tmp_dt_end_stamp_obj
		if tmp_dt_end_stamp_obj in return_dict:
			pass
		else:
			return_dict[tmp_dt_end_stamp_obj] = {'price': tmp_end_stub_price,  'volume': 0}  # 'volume': tmp_end_stub_volume
		tmp_dt_end_stamp_obj += datetime.timedelta(minutes=1)

	# 2) 앞쪽에서 값이 missing된 경우
	tmp_dt_srt_stamp_obj = FUNC_dtRect(tmp_dt_srt_stamp)
	tmp_start_stub_price = ori_dict[tmp_dt_srt_stamp]['price']
	tmp_start_stub_volume = ori_dict[tmp_dt_srt_stamp]['volume']
	tmp_end_time_obj = start_time_obj
	while tmp_end_time_obj <= tmp_dt_srt_stamp_obj:
		#tmp_end_time_obj_convert = FUNC_dtSwtich(tmp_end_time_obj)
		if tmp_end_time_obj in return_dict:
			pass
		else:
			return_dict[tmp_end_time_obj] = {'price': tmp_start_stub_price, 'volume': 0}  # 'volume': tmp_start_stub_volume

		tmp_end_time_obj += datetime.timedelta(minutes=1)

	return return_dict


def SQ_check_opDay(datetime_item):
	"""

	:param datetime_item: assumed dt object, of datetime. will be converted into dt
	:return : Action - calculates weekend and returns weekday
	"""
	# 월, 화, 수, 목, 금, 토, 일
	# 0,  1,  2,  3, 4, 5,  6
	datetime_obj = FUNC_to_dtObject(datetime_item)
	if datetime_obj.weekday() in [5, 6]:
		return False
	else:
		return True


def sqlite_capture(db_loc):

	sqlite_conTop = sqlite3.connect(db_loc)
	sqlite_curTop = sqlite_conTop.cursor()

	def wrapper(_stock_code=None, get_codes=False):

		if get_codes:
			tmp_codes__obj = sqlite_curTop.execute("SELECT name FROM sqlite_master WHERE type='table';")
			tmp_codes = tmp_codes__obj.fetchall()
			tmp_codes = [list(value)[0] for value in tmp_codes]
			return tmp_codes

		else:
			stock_code = str(_stock_code)
			assert stock_code != None

			head_string = 'SELECT * FROM '
			tmp_selected = "'" + str(stock_code) + "'"
			tmp_df = pd.read_sql(head_string + tmp_selected, sqlite_conTop, index_col=None)

			if (not tmp_df.empty) and len(tmp_df) >= int(900 * 0.99):
				rtn_df = tmp_df.loc[(tmp_df['open'] >= 5000) \
									& (np.mean(tmp_df['volume']) >= 500)
									].copy()
				if rtn_df.empty:  # 빈 데이터
					return pd.DataFrame()  # return real empty df
				else:
					## change date to datetime
					rtn_df['date'] = pd.to_datetime(rtn_df['date'], format="%Y%m%d%H%M%S")
					print(f'stock code that was selected... : {stock_code}')
					print(f'{rtn_df.head()}')
					return rtn_df
			else:  # not enough data / or no data at all
				return pd.DataFrame()  # return real empty df

	return wrapper


def sweep_day(df, stock_code, start_date, end_date,
			  hours_back=int(13), minute_forward=int(30), _type='data',
			  min_dur=int(1)):
	"""

	:param df: dataframe input
	:param start_date:
	:param end_date:
	:return: yields hashs until stopiteration exception
		=> except StopIteration:: happends at the end of the function
	"""

	assert _type in ['data', 'answer']

	tmp_dt_start = copy.deepcopy(start_date)
	tmp_dt_sweep = copy.deepcopy(start_date)
	tmp_dt_end = end_date
	return_hash = None

	print(f'begin parsing stock_code : {stock_code}')

	while tmp_dt_sweep <= tmp_dt_end:

		## get hash
		if _type == 'answer':
			return_hash = SQ__parse_answer(ori_df_whole=df,
										   dt_now__obj=tmp_dt_sweep,
										   minute_forward=minute_forward,
										   stock_code=stock_code)
		else:
			return_hash = SQ__parse_sqData(ori_df_whole=df,
										   dt_now__obj=tmp_dt_sweep,
										   hours_duration_back=hours_back,
										   stock_code=stock_code)

		yield return_hash, tmp_dt_sweep

		## right now -> update after yield keyword
		tmp_dt_sweep += datetime.timedelta(minutes=min_dur)

	print(f'ended parsing stock_code : {stock_code}')
