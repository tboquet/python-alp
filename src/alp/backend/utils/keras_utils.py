"""Utilities for the Keras abstract backend"""

import six
from keras import optimizers
from keras.utils.layer_utils import layer_from_config


def to_dict_w_opt(model):
    """Serialize a model and add the config of the optimizer and the loss.
    """
    config = dict()
    config_m = model.get_config()
    config['config'] = {
        'class_name': model.__class__.__name__,
        'config': config_m,
    }
    if hasattr(model, 'optimizer'):
        config['optimizer'] = model.optimizer.get_config()
    if hasattr(model, 'loss'):
        if type(model.loss) == dict:
            config['loss'] = dict([(k, get_function_name(v))
                                   for k, v in model.loss.items()])
        else:
            config['loss'] = get_function_name(model.loss)

    return config


def model_from_dict_w_opt(model_dict, custom_objects=None):
    """Builds a model from a serialized model using ``to_dict_w_opt`
    """
    if custom_objects is None:
        custom_objects = {}

    model = layer_from_config(model_dict['config'],
                              custom_objects=custom_objects)

    if 'optimizer' in model_dict:
        model_name = model_dict['config'].get('class_name')
        # if it has an optimizer, the model is assumed to be compiled
        loss = model_dict.get('loss')

        # if a custom loss function is passed replace it in loss
        if model_name == "Graph":
            for l in loss:
                for c in custom_objects:
                    if loss[l] == c:
                        loss[l] = custom_objects[c]
        elif model_name == "Sequential" and loss in custom_objects:
            loss = custom_objects[loss]

        optimizer_params = dict([(
            k, v) for k, v in model_dict.get('optimizer').items()])
        optimizer_name = optimizer_params.pop('name')
        optimizer = optimizers.get(optimizer_name, optimizer_params)

        if model_name == "Sequential":
            sample_weight_mode = model_dict.get('sample_weight_mode')
            model.compile(loss=loss,
                          optimizer=optimizer,
                          sample_weight_mode=sample_weight_mode)
        elif model_name == "Graph":
            sample_weight_modes = model_dict.get('sample_weight_modes', None)
            loss_weights = model_dict.get('loss_weights', None)
            model.compile(loss=loss,
                          optimizer=optimizer,
                          sample_weight_modes=sample_weight_modes,
                          loss_weights=loss_weights)
    return model


def get_function_name(o):
    """Utility function to return the model's name
    """
    if isinstance(o, six.string_types):
        return o
    else:
        return o.__name__
