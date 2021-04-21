
from django.db.models import Q

from threading import Thread
import datetime
import time


class UpbitAPI:

    def __init__(self): pass

    @staticmethod
    def request_tick_data():
        print(f'in upbit API : ', datetime.datetime.now())