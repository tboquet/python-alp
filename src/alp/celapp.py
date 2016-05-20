"""Simple celery config"""

from celery import Celery
from . import config


app = Celery(broker=config.BROKER,
             backend=config.BACKEND)
