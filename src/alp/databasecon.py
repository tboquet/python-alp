"""Model database setup

----------------------------------------------------------------------------
"""

from pymongo import MongoClient
from .config import COLLECTION_NAME
from .config import DB_NAME
from .config import HOST_ADRESS
from .config import HOST_PORT


def get_models():
    """Utility function to retrieve the collection of models

    Returns:
        the collection of models"""
    client = MongoClient(HOST_ADRESS, HOST_PORT)
    modelization = client[DB_NAME]
    return modelization[COLLECTION_NAME]
