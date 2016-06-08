"""Model database setup

----------------------------------------------------------------------------
"""

from pymongo import MongoClient
from ..dbbackend import _collection_name
from ..dbbackend import _db_name
from ..dbbackend import _host_adress
from ..dbbackend import _host_port


def get_models():
    """Utility function to retrieve the collection of models

    Returns:
        the collection of models"""
    client = MongoClient(_host_adress, _host_port)
    modelization = client[_db_name]
    return modelization[_collection_name]


def insert(full_json):
    models = get_models()
    return models.insert_one(full_json).inserted_id


def update(inserted_id, json_changes):
    models = get_models()
    dict_id = dict()
    dict_id['_id'] = inserted_id
    models.update(dict_id, json_changes)
