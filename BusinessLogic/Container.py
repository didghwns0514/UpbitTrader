if __name__ == '__main__':
	print(f'is main!')
	from Sub_APIs.GetData import *
else:
	print('is not main!')
	from BusinessLogic.Sub_APIs.GetData import *

import traceback

from transitions import Machine
from django.db.models import Q



class Container:
	"""
	#Container for each coins
	##1) get past data
	"""


	def __init__(self, CoinName):
		self._coinName = CoinName

		self._coin_curr_price = None
		self._coin_curr_volume = None

		self._coin_candle = None
		self._coin_candle_max_length = None

	def getCandleData(self):
		try:
			#self.past_data_wrapper(n_days=
			#Minute.nonasync_past_data_wrapper(self._coinName, n_days=1)
			self._coin_candle, self._coin_candle_max_length = Minute.past_data_wrapper(self._coinName)
		except Exception as e:
			print(f'error : {e}')
			traceback.print_exc()


class State:
	pass


if __name__ == '__main__':
	"""to test fetching data"""
