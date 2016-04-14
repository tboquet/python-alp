"""Adaptor for the Keras backend"""

from keras.models import model_from_json
import keras.backend as K


def build_predict_func(mod):
    """Build Keras prediction functions based on a Keras model

    Using inputs and outputs of the graph a prediction function
    (forward pass) is compiled for prediction purpose.

    Args:
        mod(keras.models): a Graph or Sequential model

    Returns:
        a Keras (Theano or Tensorflow) function
    """
    # get inputs
    if "graph" in mod.name:
        inputs = [mod.inputs[inp].input for inp in mod.input_order]
    elif "sequential" in mod.name:
        inputs = model.inputs

    # get outputs
    outputs = []
    for out in mod.output_order:
        outputs.append(mod.outputs[out].get_output(train=False))

    return K.function(inputs, outputs, updates=mod.state_updates)


def train_model(model_str, custom_objects, datas, datas_val, batch_size,
                nb_epoch, callbacks):
    """Train a model given hyperparameters and a serialized model"""

    loss = []
    val_loss = []
    # load model
    model = model_from_json(model_str, custom_objects=custom_objects)

    # fit the model according to the input/output type
    if model.name == 'Graph':
        for d, dv in zip(datas, datas_val):
            h = model.fit(data=d,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=dv)
            loss += h.history['loss']
            if 'val_loss' in h.history:
                val_loss += h.history['val_loss']
    elif model.name == 'Sequential':
        # unpack data
        for d, dv in zip(datas, datas_val):
            X, y = d['X'], d['y']
            X_val, y_val = dv['X'], dv['y']
            h = model.fit(X=X,
                            y=y,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(X_val, y_val))
            loss += h.history['loss']
            if 'val_loss' in h.history:
                val_loss += h.history['val_loss']
    else:
        raise NotImplementedError('This type of model is not supported')

    return loss, val_loss, model

