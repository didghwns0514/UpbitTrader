from django.apps import AppConfig


class UpbitConfig(AppConfig):
    print(f'UpbitConfig is up!')
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Upbit'

    def ready(self): # on module ready
        from BusinessLogic import BL_Scheduler
        BL_Scheduler.start_upbit()