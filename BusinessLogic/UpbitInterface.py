
from django.db.models import Q

from threading import Thread
import datetime
import time

"""
https://wikidocs.net/31063
https://wikidocs.net/117438

"""

class UpbitAPI:

    def __init__(self):
        var_check_class = 'Upbit-api'

    @staticmethod
    def request_tick_data():
        print(f'in upbit API : {UpbitAPI.var_check_class}', datetime.datetime.now())