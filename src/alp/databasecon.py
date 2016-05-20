"""Model database setup

----------------------------------------------------------------------------
"""

from pymongo import MongoClient
from . import config


def get_models():
    """Utility function to retrieve the collection of models

    Returns:
        the collection of models"""
    client = MongoClient(config.HOST_ADRESS, config.HOST_PORT)
    modelization = client[config.DB_NAME]
    return modelization[config.COLLECTION_NAME]
