"""
Celery config
=============
"""

from celery import Celery
from . import appcom as apc


RESULT_SERIALIZER = 'json'

app = Celery(broker=apc._broker,
             backend=apc._backend)

app.conf.update(task_serializer='pickle',
                result_serializer=RESULT_SERIALIZER,
                accept_content=['pickle', 'json'])
