import pandas as pd
import datetime
import traceback

from .Async_module import async_wrapper
from transitions import Machine
from django.db.models import Q

import pyupbit



class Container:
	"""
	#Container for each coins
	##1) get past data
	"""


	def __init__(self, CoinName):
		self._coinName = CoinName

		self._coin_curr_price = None
		self._coin_curr_volume = None

		try:
			self.past_data_wrapper(n_days=1)
		except Exception as e:
			print(f'error : {e}')
			traceback.print_exc()


	def past_data_wrapper(self, n_days=2, count=200, to=datetime.datetime.now()):
		"""wrapper for get_pas_data function"""
		request_num = int(n_days) * 24 * 60
		iteration= range(request_num//count + 1 )
		arg_function = [ self.get_past_data for _ in iteration]
		arg_param = [[count, to - datetime.timedelta(minutes=200*i)] for i in iteration ]

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


	def get_past_data(self, _count, _to):
		""" get minute 2days data"""
		df = pyupbit.get_ohlcv(
			ticker=self._coinName,
			interval="minute1",
			count=_count,
			to = _to
		)


		return df


class State:
	pass