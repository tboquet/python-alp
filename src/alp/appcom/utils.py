"""Utility functions for the appcom module

----------------------------------------------------------------------------
"""


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


def init_backend(backend):
    """Initialization of the backend

    Args:
        backend(str): only 'keras' at the moment

    Returns:
        the backend, the backend name and the backend version
    """
    if backend == 'keras':
        from ..backend import keras_backend as ABE
        backend = ABE
        backend_version = None
        backend_m = ABE.get_backend()
        backend_name = backend_m.__name__
        if hasattr(backend_m, '__version__'):
            backend_version = backend_m.__version__
    return backend, backend_name, backend_version


def switch_backend(backend_name):
    if backend_name == 'keras':
        from ..backend.keras_backend import get_backend
    elif backend_name == 'sklearn':
        from ..backend.keras_backend import get_backend
    return get_backend()
