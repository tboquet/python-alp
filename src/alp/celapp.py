"""
Celery config
=============
"""

from celery import Celery
from . import appcom as apc


app = Celery(broker=apc._broker,
             backend=apc._backend)

app.conf.update(task_serializer='pickle',
                result_serializer='json',
                accept_content=['pickle', 'json'])
