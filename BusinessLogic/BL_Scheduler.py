from apscheduler.schedulers.background import BackgroundScheduler
from .UpbitInterface import UpbitAPI

"""
https://tomining.tistory.com/138
"""
def start_upbit():
    scheduler = BackgroundScheduler()
    scheduler.add_job(UpbitAPI.request_tick_data, 'interval', seconds=1)

    scheduler.start()