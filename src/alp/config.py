"""Configuration file for the application"""
import os


# MongoDB config

HOST_ADRESS = 'mongo_m'
HOST_PORT = 27017
DB_NAME = 'modelization'
COLLECTION_NAME = 'models'

# Rabbitmq config

BROKER = 'amqp://guest:guest@rabbitmq:5672//'
BACKEND = 'mongodb://mongo_r:27017'

if os.getenv("TEST_MODE") == "ON":
    HOST_ADRESS = '127.0.0.1'
    BACKEND = 'mongodb://127.0.0.1:27017'
