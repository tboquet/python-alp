import copy
import hashlib
import json
import pickle
import numpy as np

from datetime import datetime


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


def create_gen_hash(gen):
    """Creates a hash based on the data passed

    The unique descriptors are based on the mean of the arrays passed and the
    sum of all the elements of the first lines of the first axis.

    Args:
        data(list): a dictionnary of the model

    Returns:
        a md5 hash of the data"""
    pickle_gen = pickle.dumps(gen)
    dh = hashlib.md5()
    str_concat_g = str(pickle_gen)
    dh.update(str_concat_g.encode('utf-8'))
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


def make_all_hash(model_c, batch_size, data_hash, _path_h5):
    hexdi_m = create_model_hash(model_c, batch_size)
    params_dump = create_param_dump(_path_h5, hexdi_m, data_hash)
    return hexdi_m, params_dump


def open_dataset_gen(generator):
    if hasattr(generator, 'data_stream'):
        data_stream = generator.data_stream
        data_stream.dataset.open()
    elif hasattr(generator, 'dataset'):
        generator.dataset.open()
    else:
        raise NotImplementedError('not able to open the dataset')


def transform_gen(gen_train, mod_name):
    """Transform generators of tupple to generators of dicts

    Args:
        gen_train(Fuel data stream): a fuel training data generator
        gen_val(Fuel data stream): a fuel validation data generator

    Yield:
        a dictionnary mapping training and testing data to numpy arrays"""
    names_dict = gen_train.sources

    inp = 'input_'
    out = 'output_'

    li = 'list'

    list_outputs = False
    list_inputs = False

    open_dataset_gen(gen_train)

    while 1:
        for d in gen_train.get_epoch_iterator():
            data = zip(d, names_dict)
            inputs_list = []
            outputs_list = []
            inputs_dict = dict()
            outputs_dict = dict()
            for arr, name in data:
                if inp in name:
                    if li in name:
                        inputs_list.append(arr)
                        list_inputs = True
                    else:
                        inputs_dict[name[6:]] = arr
                elif out in name:
                    if li in name:
                        outputs_list.append(arr)
                        list_outputs = True
                    else:
                        outputs_dict[name[7:]] = arr
                else:  # pragma: no cover
                    raise("Not input nor output, please check your generator")
            fin_outputs = outputs_dict
            fin_inputs = inputs_dict

            if list_outputs:
                fin_outputs = outputs_list
            if list_inputs:
                fin_inputs = inputs_list
            data_out = (fin_inputs, fin_outputs)
            if mod_name == 'Graph':
                fin_inputs.update(fin_outputs)
                data_out = fin_inputs
            yield data_out


def train_pipe(train_f, save_f, model, data, data_val, generator, params_dump,
               data_hash, hexdi_m,
               *args, **kwargs):
    results, model = train_f(model['model_arch'], data,
                             data_val,
                             generator=generator,
                             *args, **kwargs)
    res_dict = {
        'iter_stopped': results['metrics']['iter'],
        'trained': 1,
        'date_finished_training': datetime.now()}
    for metric in results['metrics']:
        res_dict[metric] = results['metrics'][metric]
        if metric in ['loss', 'val_loss']:
            res_dict[metric] = np.min(results['metrics'][metric])

    save_f(model, params_dump)
    results['model_id'] = hexdi_m
    results['data_id'] = data_hash
    results['params_dump'] = params_dump
    return results, res_dict
