from __future__ import absolute_import
from __future__ import print_function

import json
import os


_alp_base_dir = os.path.expanduser('~')
if not os.access(_alp_base_dir, os.W_OK):  # pragma: no cover
    _alp_base_dir = '/tmp'


_alp_dir = os.path.join(_alp_base_dir, '.alp')
if not os.path.exists(_alp_dir):
    os.makedirs(_alp_dir)

_db_engine = 'mongodb'
_host_adress = 'mongo_m'
_host_port = 27017
_db_name = 'modelization'
_collection_name = 'models'

if os.getenv("TEST_MODE") == "ON":
    _host_adress = '127.0.0.1'

# note: we have to be able to accept other structures

_config_path = os.path.expanduser(os.path.join(_alp_dir, 'alpdb.json'))
if os.path.exists(_config_path):  # pragma: no cover
    _config = json.load(open(_config_path))
    _db_engine = _config.get('db_engine', 'mongodb')
    assert _db_engine in {'mongodb'}
    _host_adress = _config.get('host_adress', 'mongo_m')
    _host_port = _config.get('host_port', 27017)
    _db_name = _config.get('db_name', 'modelization')
    _collection_name = _config.get('collection_name', 'models')

# save config file
_config = {'_db_engine': _db_engine,
           '_host_adress': _host_adress,
           '_host_port': _host_port,
           '_db_name': _db_name,
           '_collection_name': _collection_name}

with open(_config_path, 'w') as f:
    f.write(json.dumps(_config, indent=4))

# import backend
if _db_engine == 'mongodb':
    from ..dbbackend.mongo_backend import *  # NOQA
else:
    raise Exception('Unknown backend: ' + str(_db_engine))
