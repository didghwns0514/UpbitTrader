if __name__ == '__main__':
	from ..Async_module import async_wrapper
else:
	try:
		from BusinessLogic.Async_module import async_wrapper
	except ImportError:
		from ..Async_module import async_wrapper
	finally:
		import hashlib
		from urllib.parse import urlencode
		import jwt
		import requests
		import uuid

import pandas as pd
import datetime
import time
import pyupbit

class Minute:

	def __init__(self): pass

	@staticmethod
	def nonasync_past_data_wrapper( _coinName, n_days=2, count=200, to=datetime.datetime.now()):
		_type = 2
		request_num = int(n_days) * 24 * 60
		#print(f'request_num//count : {request_num//count}, request_num % count : {request_num % count} ')
		iteration= range( (request_num//count) + 1 + (0 if request_num % count == 0 else 1 ))

		arg_param = [ (to - datetime.timedelta(minutes=200*i)).strftime("%Y-%m-%d %H:%M:%S") for i in iteration]
		#print(f'arg_param : {arg_param}')


		for time_value in arg_param:
			if _type == 1:
				df = Minute.get_past_data(_count=count, _to=time_value)
				if isinstance(df, pd.DataFrame):
					print('True')
				else:
					print('False')
				time.sleep(0.01)
			elif _type == 2:
				rtn = Minute.api_get_pas_data(_coinName, _count=count, _to=time_value)
				print(rtn)

	@staticmethod
	def past_data_wrapper(n_days=2, count=200, to=datetime.datetime.now()):
		"""wrapper for get_pas_data function"""
		request_num = int(n_days) * 24 * 60
		iteration= range(request_num//count + 1 )
		arg_function = [ Minute.get_past_data for _ in iteration]
		arg_param = [[count, (to - datetime.timedelta(minutes=200*i)).strftime("%m-%d-%Y %H:%M:%S")] for i in iteration ]

		#print(f'arg_function : {arg_function}, arg_param : {arg_param}')

		df_container = async_wrapper(arg_function, arg_param)
		#print(f'df_container : {df_container}')
		df_container = [df for df in df_container if isinstance(df, pd.DataFrame)]

		df_checker = [isinstance(df, pd.DataFrame) for df in df_container  ]
		print(f'df_checker : {df_checker}')
		tmp_concat = None
		if not df_container:
			return
		else: # has more than one
			tmp_concat = df_container[0]
			for i, tmp_df in enumerate(df_container):
				if i >= 1:
					tmp_concat = pd.concat([tmp_concat, tmp_df])
		#print(f'tmp_concat : {tmp_concat}')
		print(f'\n'*2)
		return tmp_concat

	@staticmethod
	def get_past_data( _coinName, _count, _to):
		""" get minute 2days data"""

		df = pyupbit.get_ohlcv(
			ticker=_coinName,
			interval="minute1",
			count=_count,
			to = _to
		)

		return df

	@staticmethod
	def api_get_pas_data(coin_name, _count, _to):
		#url = "https://api.upbit.com/v1/candles/minutes/1"
		url = "https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/1"
		queryString = {
			"market" : 'CRIX.UPBIT.'+coin_name,
			"count" : _count,
			"to" : _to
		}
		response = requests.request("GET", url, params=queryString)
		return response.json()[0]