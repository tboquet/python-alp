"""Model database setup

----------------------------------------------------------------------------
"""

from pymongo import DESCENDING
from pymongo import MongoClient
from ..dbbackend import _models_collection
from ..dbbackend import _generators_collection
from ..dbbackend import _db_name
from ..dbbackend import _host_adress
from ..dbbackend import _host_port


def get_models():
    """Utility function to retrieve the collection of models

    Returns:
        the collection of models"""
    client = MongoClient(_host_adress, _host_port)
    modelization = client[_db_name]
    return modelization[_models_collection]


def get_generators():
    """Utility function to retrieve the collection of generators

    Returns:
        the collection of generators"""
    client = MongoClient(_host_adress, _host_port)
    modelization = client[_db_name]
    return modelization[_generators_collection]


def insert(full_json, collection, upsert=False):
    """Insert an observation in the db

    Args:
        full_json(dict): a dictionnary mapping variable names to
            carateristics of the model

    Returns:
        the id of the inserted object in the db"""
    filter_db = dict()
    filter_db['mod_data_id'] = full_json['mod_data_id']
    doc_id = collection.find_one(filter_db)
    if doc_id is not None:
        doc_id = doc_id['_id']
    if upsert is True:
        inserted = collection.find_one_and_update(
            filter_db, {'$set': full_json}, upsert=upsert)
    else:
        inserted = collection.insert_one(full_json).inserted_id
    return inserted


def update(inserted_id, json_changes):
    """Update an observation in the db

    Args:
        insert_id(int): the id of the observation
        json_changes(dict): the changes to do in the db"""
    models = get_models()
    dict_id = dict()
    dict_id['_id'] = inserted_id
    models.update(dict_id, json_changes)


def create_db(drop=True):
    """Delete (and optionnaly drop) the modelization database and collection"""
    client = MongoClient(_host_adress, _host_port)
    modelization = client[_db_name]
    if drop:
        modelization.drop_collection(_models_collection)
    models = modelization['models']
    return models.create_index([('mod_data_id', DESCENDING)],
                               unique=True)
