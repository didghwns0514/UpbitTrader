#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# check dev / deploy
env_string = 'config.settings.deploy' \
    if not bool(os.getenv('IS_DEVELOP', True)) \
    else 'config.settings.development'
print(f'env_string : {env_string}')
print(f'bool(os.getenv("IS_DEVELOP", True)) : ', bool(os.getenv('IS_DEVELOP', True)))
print(f"os.getenv('IS_DEVELOP') : {os.getenv('IS_DEVELOP')}")
def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', env_string)
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
