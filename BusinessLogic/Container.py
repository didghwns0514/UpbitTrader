import pandas as pd
import datetime
import traceback

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

		df_concat = self.get_past_data(count, to)

		while isinstance(df_concat, pd.DataFrame) and request_num >= 0 :
			to = to - datetime.timedelta(minutes=200)
			tmp_df = self.get_past_data(count, to)
			if isinstance(tmp_df, pd.DataFrame): #
				df_concat = pd.concat([df_concat, tmp_df])
				request_num -= count
			else:
				break
		if isinstance(df_concat, pd.DataFrame):
			print(f'df_concat head : {df_concat.head(10)}')
			print(f'df_concat tail : {df_concat.tail(10)}')
			print(f'df_concat length : {len(df_concat)}')

		else:
			print(f'error!')

		print(f'\n'*2)
		return df_concat

	def get_past_data(self, _count, _to):
		""" get minute 2days data"""
		df = pyupbit.get_ohlcv(
			ticker=self._coinName,
			interval="minute1",
			count=_count,
			to = _to
		)
		# print(f'_to in data req : {_to}')
		# if isinstance(df, pd.DataFrame):
		# 	print(f'name : {self._coinName} ')
		# 	print(f'df : {df}')
		# else:
		# 	print(f'error......!')
		# 	print(f'name : {self._coinName} ')
		# 	print(f'df : {df}')

		return df


class State:
	pass