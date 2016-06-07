"""Model database setup

----------------------------------------------------------------------------
"""

from pymongo import MongoClient
from . import appcom as apc


def get_models():
    """Utility function to retrieve the collection of models

    Returns:
        the collection of models"""
    client = MongoClient(apc._host_adress, apc._host_port)
    modelization = client[apc._db_NAME]
    return modelization[apc._collection_name]


def insert(full_json):
    models = get_models()
    return models.insert_one(full_json).inserted_id

def update(inserted_id, json_changes):
    models = get_models()
    dict_id = dict()
    dict_id['_id'] = inserted_id
    models.update(dict_id, json_changes)
