"""Utility functions for the appcom module

----------------------------------------------------------------------------
"""

import functools
import pickle
import threading
from itertools import islice

from six.moves import zip as szip


def _get_backend_attributes(ABE):
    """Gets the backend attributes.

    Args:
        ABE(module): the module to get attributes from.

    Returns:
        the backend, the backend name and the backend version

    """
    backend_m = ABE.get_backend()
    backend_name = backend_m.__name__
    if hasattr(backend_m, '__version__'):
        backend_version = backend_m.__version__
    else:
        backend_version = None

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
            "this backend is not supported: {}".format(
                model))  # pragma: no cover

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
    else:
        raise NotImplementedError
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


def norm_iterator(iterable):
    """returns a normalized iterable of tuples"""
    if isinstance(iterable, list):
        names = ['list_' + str(i) for i, j in enumerate(iterable)]
        return szip(names, iterable)
    elif isinstance(iterable, dict):
        return iterable.items()
    else:
        raise NotImplementedError('Iterables other than lists and dicts '
                                  'cannot be passed to this function')


def window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable"""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:  # pragma: no cover
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def to_fuel_h5(inputs, outputs, slices, names,
               file_name, file_path=''):
    """Transforms list of numpy arrays to a structured hdf5 file


    Args:
        inputs(list): a list of inputs(numpy.arrays)
        outputs(list): a list of outputs(numpy.arrays)
        slices(list): a list of int representing the end of a slice and the
            begining of another slice. The last slice is automatically added
            if missing (maximum length of the inputs).
        names(list): a list of names for the datasets
        file_name(str): the name of the file to save.
        file_path(str): the path where the file is located

    Returns:
        The file full path
    """
    import h5py
    import os
    from fuel.datasets.hdf5 import H5PYDataset

    suffix = 'hdf5'

    inp = 'input_'
    out = 'output_'

    full_path = os.path.join(file_path, file_name.lower() + '.' + suffix)
    f = h5py.File(full_path, mode='w')

    dict_data_set = dict()
    split_dict = dict()
    for name in names:
        split_dict[name] = dict()

    slices.append(max_v_len(inputs))

    def insert_info_h5(iterable, suf):
        names_out = []
        for k, v in norm_iterator(iterable):
            dict_data_set[suf + k] = f.create_dataset(suf + k, v.shape,
                                                      v.dtype)
            dict_data_set[suf + k][...] = v
            for sl, name in zip(window(slices, 2), names):
                split_dict[name][suf + k] = sl
            names_out.append(suf + str(k))
        return names_out

    inputs_names = insert_info_h5(inputs, inp)
    outputs_names = insert_info_h5(outputs, out)

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()
    return full_path, inputs_names, outputs_names


def max_v_len(iterable_to_check):
    """Returns the max length of a list of iterable"""
    max_v = 0
    for _, v in norm_iterator(iterable_to_check):
        if len(v) > max_v:  # pragma: no cover
            max_v = len(v)
    return max_v


def pickle_gen(gen_train, data_val):
    """Check and serialize the validation data object and serialize the
    training data generator.

    Args:
        gen_train(generator): the training data generator
        data_val(dict or generator): the training data object

    Returns:
        normalized datasets"""
    gen_train = [pickle.dumps(g).decode('raw_unicode_escape')
                 for g in gen_train]

    val_gen = check_gen(data_val)

    if val_gen:
        data_val = [pickle.dumps(g).decode('raw_unicode_escape')
                    for g in data_val]
    return gen_train, data_val


def check_gen(iterable):
    """Check if the last object of the iterable is an iterator

    Args:
        iterable(list): a list containing data.

    Returns:
        True if the last object is a generator, False otherwise.
        """
    is_gen = (hasattr(iterable[-1], 'next') or
              hasattr(iterable[-1], '__next__'))
    is_gen += 'fuel' in repr(iterable[-1])

    return is_gen


def get_nb_chunks(generator):
    """Get the number of chunks that yields a generator

    Args:
        generator: a Fuel generator

    Returns:
        number of chunks (int)"""
    if hasattr(generator, 'iteration_scheme'):
        if generator.iteration_scheme is not None:
            batch_size = generator.iteration_scheme.batch_size
            nb_examples = len(generator.iteration_scheme.indices)
            return nb_examples // batch_size
        else:
            if hasattr(generator, 'data_stream'):
                return get_nb_chunks(generator.data_stream)
            else:
                raise Exception('No data stream in the generator')
