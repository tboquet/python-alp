import hashlib
import marshal


def create_model_hash(model, batch_size):
    # convert dict to json string
    if 'ser_metrics' in model:
        model.pop('ser_metrics')
    if 'metrics' in model:
        model.pop('metrics')
    model_str = marshal.dumps(model)

    # create the model hash from the stringified json
    mh = hashlib.md5()
    str_concat_m = str(model_str) + str(batch_size)
    mh.update(str_concat_m.encode('utf-8'))
    return mh.hexdigest()


def create_data_hash(data):
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
    return _path_h5 + hexdi_m + hexdi_d + '.h5'
