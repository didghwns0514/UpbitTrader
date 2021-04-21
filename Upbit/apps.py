from django.apps import AppConfig

# class TaskThread(Thread):
#
#     def __init__(self):
#         Thread.__init__(self)
#
#     # override run method
#     def run(self):
#         print('Thread running : ', datetime.datetime.now())
#         #time.sleep(1)

class UpbitConfig(AppConfig):
    print(f'UpbitConfig is up!')
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Upbit'

    def ready(self): # on module ready
        from BusinessLogic import BL_Scheduler
        BL_Scheduler.start_upbit()