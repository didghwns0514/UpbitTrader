from .base import * # import all from base.py but diff is made under this line
import os
import myconfig

SECRET_KEY = os.environ.get(
	'DJANGO_SECRET_KEY',
	myconfig.SE_SECRET_KEY)
DATABASES = {
	'default': {
		'ENGINE': 'django.db.backends.mysql', #1
		'NAME': 'upbit', #2
		'USER': os.environ.get('DATABASES_USER',myconfig.SE_DATABASES_USER), #3
		'PASSWORD': os.environ.get('DATABASES_PSWD', myconfig.SE_DATABASES_PSWD),  #4
		'HOST': os.environ.get('DATABASE_URL', myconfig.SE_DATABASE_URL),   #5
		'PORT': '3306', #6
	}
}
SE_DEBUG = bool(os.environ.get('DJANGO_DEBUG', True))

ALLOWED_HOSTS = ['*']