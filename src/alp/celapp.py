"""Simple celery config"""

from celery import Celery
from .config import BACKEND
from .config import BROKER


app = Celery(BROKER,
             BACKEND)
