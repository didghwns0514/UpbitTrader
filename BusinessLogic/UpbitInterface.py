
from .Container import Container
from threading import Thread

import datetime
import time
import os
import pyupbit

if os.getenv('IS_DEVELOP') != 'False':
    from config.settings.myconfig import Q_SECRET_KEY, Q_ACCESS_KEY

_Q_ACCESS_KEY = Q_ACCESS_KEY if not os.getenv('Q_ACCESS_KEY') else os.getenv('Q_ACCESS_KEY')
_Q_SECRET_KEY = Q_SECRET_KEY if not os.getenv('Q_SECRET_KEY') else os.getenv('Q_SECRET_KEY')

"""
https://wikidocs.net/31063
https://wikidocs.net/117438

"""
class Upbit:

    def __init__(self, marketName="KRW"):
        self._marketName = marketName
        self._tickers = pyupbit.get_tickers(fiat=self._marketName)
        print(f'total tickers : {len(self._tickers)}')
        self._lookup = {}

        self.createContainers()

    def createContainers(self):

        for coin_name in self._tickers:
            if coin_name not in self._lookup:
                self._lookup[coin_name] = Container(coin_name)


class UpbitInterface:

    var_check_class = 'Upbit-api'
    MARKET_NAME = "KRW"
    UB = Upbit(MARKET_NAME)

    @staticmethod
    def request_tick_data():
        print(f'in upbit API : {UpbitInterface.var_check_class}', datetime.datetime.now())


