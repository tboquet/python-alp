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
    """Insert an observation in the db

    Args:
        full_json(dict): a dictionnary mapping variable names to
            carateristics of your model

    Returns:
        the id of the inserted object in the db"""
    models = get_models()
    return models.insert_one(full_json).inserted_id


def update(inserted_id, json_changes):
    """Update an observation in the db

    Args:
        insert_id(int): the id of the observation
        json_changes(dict): the changes to do in the db"""
    models = get_models()
    dict_id = dict()
    dict_id['_id'] = inserted_id
    models.update(dict_id, json_changes)
