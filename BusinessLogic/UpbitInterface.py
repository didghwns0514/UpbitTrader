if __name__ == '__main__':
    from Container import Container
else:
    from .Container import Container

from threading import Thread
import datetime
import time
import os
import pyupbit
import pandas as pd


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
        self._lookup = {}

        self.createContainers()
        self.getCandleData()
        self.analysisCandleData()

    def createContainers(self):

        for coin_name in self._tickers:
            if coin_name not in self._lookup:
                self._lookup[coin_name] = Container(coin_name)


    def getCandleData(self):

        for coin_name in self._lookup:
            self._lookup[coin_name].getCandleData()


    def analysisCandleData(self):
        cnt = 0
        for coin_name in self._lookup:
            if isinstance(self._lookup[coin_name]._coin_candle, pd.DataFrame):
                cnt += 1
                tmp_ratio = round(len(self._lookup[coin_name]._coin_candle) / self._lookup[coin_name]._coin_candle_max_length, 2)
                print(tmp_ratio)
        print(f'total df : {cnt}')



class UpbitInterface:

    var_check_class = 'Upbit-api'
    MARKET_NAME = "KRW"
    UB = Upbit(MARKET_NAME)

    @staticmethod
    def request_tick_data():
        print(f'in upbit API : {UpbitInterface.var_check_class}', datetime.datetime.now())


if __name__ == '__main__':
    """to test fetching data"""
    tmp_obj = UpbitInterface()