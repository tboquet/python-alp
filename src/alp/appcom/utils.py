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


def init_backend(backend):
    """Initialization of the backend

    Args:
        backend(str): only 'keras' or 'sklearn' at the moment

    Returns:
        the backend, the backend name and the backend version
    """
    if backend == 'keras':
        from ..backend import keras_backend as ABE
    elif backend == 'sklearn':
        from ..backend import sklearn_backend as ABE
    else:
        raise NotImplementedError(
            "this backend is not supported")  # pragma: no cover

    return _get_backend_attributes(ABE)


def switch_backend(backend_name):
    if backend_name == 'keras':
        from ..backend.keras_backend import get_backend
    elif backend_name == 'sklearn':
        from ..backend.keras_backend import get_backend
    return get_backend()


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
