from __future__ import absolute_import
from __future__ import print_function

import json
import os

from .core import *

_alp_base_dir = os.path.expanduser('~')
if not os.access(_alp_base_dir, os.W_OK):  # pragma: no cover
    _alp_base_dir = '/tmp'


_alp_dir = os.path.join(_alp_base_dir, '.alp')
if not os.path.exists(_alp_dir):  # pragma: no cover
    os.makedirs(_alp_dir)

# Defaults

# App config
_broker = 'amqp://guest:guest@rabbitmq:5672//'
_backend = 'mongodb://mongo_r:27017'

# Parameters
_path_h5 = '/parameters_h5/'

if os.getenv("TEST_MODE") == "ON":  # pragma: no cover
    _backend = 'mongodb://127.0.0.1:27018'
    _path_h5 = ''
    _broker = 'amqp://guest:guest@localhost:5672//'

elif os.getenv("WORKER") == "TRUE":  # pragma: no cover
    _backend = 'mongodb://mongo_r:27017'  # pragma: no cover
_config_path = os.path.expanduser(os.path.join(_alp_dir, 'alpapp.json'))

if os.path.exists(_config_path):  # pragma: no cover
    _config = json.load(open(_config_path))
    _broker = _config.get('broker', 'amqp://guest:guest@rabbitmq:5672//')
    _backend = _config.get('backend', 'mongodb://mongo_r:27017')
    _path_h5 = _config.get('path_h5', '/parameters_h5/')

# save config file
_config = {'_broker': _broker,
           '_backend': _backend,
           '_path_h5': _path_h5}

with open(_config_path, 'w') as f:
    f.write(json.dumps(_config, indent=4))

__all__ = ["Experiment"]
