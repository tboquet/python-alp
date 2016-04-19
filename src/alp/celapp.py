"""Simple celery config"""

from celery import Celery

app = Celery(broker='amqp://guest:guest@rabbitmq:5672//',
             backend='mongodb://mongo_r:27017')
