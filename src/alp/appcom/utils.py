"""Utility functions for the appcom module

----------------------------------------------------------------------------
"""

import functools
import threading


def sliced(data, nb_train, nb_test, offset):
    """Given a dataset, returns indexes to split the data into
    a train and a validation set.

    Args:
        data(dict): a dictionnary mapping names to np.arrays
        nb_train(int): the number of train samples
        nb_test(int): the number of test samples
        offset(int): the first observation offset

    Returns:
        `beg`, `endt`, `endv`, the indexes corresponding to
         the beginning, the end of training end the end of
         testing."""
    for k in data.keys():
        first = k
        break
    assert len(data[first]) > nb_train + nb_test + offset, \
        'nb or nb + offset too large:' \
        ' len(data):{}' \
        ', len(selection): {}'.format(len(data[first]),
                                      nb_train + nb_test + offset)
    beg = offset - len(data[first])
    endt = beg + nb_train
    endv = endt + nb_test
    return beg, endt, endv


def _get_backend_attributes(ABE):
    """Gets the backend attributes.

    Args:
        ABE(module): the module to get attributes from.

    Returns:
        the backend, the backend name and the backend version

    """
    backend_version = None
    backend_m = ABE.get_backend()
    backend_name = backend_m.__name__
    if hasattr(backend_m, '__version__'):
        backend_version = backend_m.__version__

    return ABE, backend_name, backend_version


def init_backend(model):
    """Initialization of the backend

    Args:
        backend(str): only 'keras' or 'sklearn' at the moment

    Returns:
        the backend, the backend name and the backend version
    """
    if 'keras' in repr(model):
        from ..backend import keras_backend as ABE
    elif 'sklearn' in repr(type(model)):
        from ..backend import sklearn_backend as ABE
    else:
        raise NotImplementedError(
            "this backend is not supported")  # pragma: no cover

    return _get_backend_attributes(ABE)


def switch_backend(backend_name):
    """Switch the backend based on it's name

    Args:
        backend_name(str): the name of the backend to import

    Return:
        the backend asked"""
    if backend_name == 'keras':
        from ..backend.keras_backend import get_backend
    elif backend_name == 'sklearn':
        from ..backend.keras_backend import get_backend
    return get_backend()


def list_to_dict(list_to_transform):
    """Transform a list of object to a dict

    Args:
        list_to_transform(list): the list to transform

    Returns:
        a dictionnary mapping names of the objects to objects"""
    return {el.__name__: el for el in list_to_transform}


def background(f):
    '''
    a threading decorator
    use @background above the function you want to run in the background
    '''
    @functools.wraps(f)
    def bg_f(*a, **kw):
        t = threading.Thread(target=f, args=a, kwargs=kw)
        t.start()
        return t
    return bg_f


def imports(packages=None):
    """A decorator to import packages only once when a function is serialized

    Args:
        packages(list or dict): a list or dict of packages to import. If the
            object is a dict, the name of the import is the key and the value
            is the module. If the object is a list, it's transformed to a dict
            mapping the name of the module to the imported module.
    """
    if packages is None:
        packages = dict()

    def dec(wrapped):
        @functools.wraps(wrapped)
        def inner(*args, **kwargs):
            packs = packages
            if isinstance(packages, list):
                packs = list_to_dict(packages)
            for name, pack in packs.items():
                if name not in wrapped.__globals__:
                    wrapped.__globals__[name] = pack
            return wrapped(*args, **kwargs)
        return inner
    return dec
