"""Utility functions for the appcom module

----------------------------------------------------------------------------
"""
from functools import wraps


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


def appbackend(f):
    """A simple decorator to run a function using the correct backend"""
    @wraps(f)
    def f_async(mod, *args, **kwargs):
        """Wrapped function

        Returns:
            the same function where we load the correct backend
        """
        if "ABE" not in f.__globals__:
            if mod['backend'] == 'keras':
                from ..backend import keras_backend as ABE

            f.__globals__['ABE'] = ABE
        return f(mod, *args, **kwargs)
    return f_async
