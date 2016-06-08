"""Simple celery config"""

from celery import Celery
from . import appcom as apc


app = Celery(broker=apc._broker,
             backend=apc._backend)
