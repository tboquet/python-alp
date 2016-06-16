import copy
import hashlib
import json


def clean_model(model):
    """Clean a dict of a model of uncessary elements

    Args:
        model(dict): a dictionnary of the model

    Returns:
        a new cleaned dict"""
    model_c = copy.deepcopy(model)
    if 'ser_metrics' in model_c['model_arch']:
        model_c['model_arch'].pop('ser_metrics')
    if 'metrics' in model_c['model_arch']:
        model_c['model_arch'].pop('metrics')
    return model_c


def create_model_hash(model, batch_size):
    """Creates a hash based on the dict of a model and the batch size

    Args:
        model(dict): a dictionnary of the model
        batch_size(int): the batch size

    Returns:
        a md5 hash of the model"""
    # convert dict to json string
    model_str = json.dumps(model)

    # create the model hash from the stringified json
    mh = hashlib.md5()
    str_concat_m = str(model_str) + str(batch_size)
    mh.update(str_concat_m.encode('utf-8'))
    return mh.hexdigest()


def create_data_hash(data):
    """Creates a hash based on the data passed

    The unique descriptors are based on the mean of the arrays passed and the
    sum of all the elements of the first lines of the first axis.

    Args:
        data(list): a dictionnary of the model

    Returns:
        a md5 hash of the data"""
    un_data_m = 0
    un_data_f = 0
    for i, _ in enumerate(data):
        for key in data[i]:
            un_data_m += data[i][key].mean()
            un_data_f += data[i][key][0].sum()

    dh = hashlib.md5()
    str_concat_d = str(un_data_m) + str(un_data_f)
    dh.update(str_concat_d.encode('utf-8'))
    return dh.hexdigest()


def create_param_dump(_path_h5, hexdi_m, hexdi_d):
    """Create a the path where to dump the params

    Args:
        _path_h5(str): the base path
        hexdi_m(str): the model hash
        hexdi_d(str): the data hash

    Returns:
        the full path where to dump the params"""
    return _path_h5 + hexdi_m + hexdi_d + '.h5'
