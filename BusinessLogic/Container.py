if __name__ == '__main__':
	from Sub_APIs.GetData import *
else:
	from .Sub_APIs.GetData import *

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

		try:
			#self.past_data_wrapper(n_days=
			self.nonasync_past_data_wrapper(n_days=1)
		except Exception as e:
			print(f'error : {e}')
			traceback.print_exc()




class State:
	pass


if __name__ == '__main__':
	"""to test fetching data"""
