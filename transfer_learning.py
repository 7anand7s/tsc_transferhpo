import numpy as np
import sklearn
import time
import json
import tensorflow as tf
from inception import _inception_module, _shortcut_layer
from utils.utils import transform_labels, calculate_metrics, save_test_duration, save_logs


def prepare_data(datasets_dict, dataset_name):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)
    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc


def train(pre_model=None, config=None, datasets_dict=None, dataset_name=None,
          dataset_name_tranfer=None, file_path=None, callbacks=None, write_output_dir=None):

    # read train, val and test sets
    x_train = datasets_dict[dataset_name_tranfer][0]
    y_train = datasets_dict[dataset_name_tranfer][1]

    y_true_val = None
    y_pred_val = None

    x_test = datasets_dict[dataset_name_tranfer][-2]
    y_test = datasets_dict[dataset_name_tranfer][-1]

    batch_size = config['batch_size'] if config['batch_size'] else 64
    if batch_size is None:
        mini_batch_size = int(min(x_train.shape[0] / 10, 16))
    else:
        mini_batch_size = batch_size

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)

    # transform the labels from integers to one hot vectors
    y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    start_time = time.time()
    # remove last layer to replace with a new one
    # input_shape = (None, x_train.shape[2])
    input_shape = x_train.shape[1:]
    model = build_model(input_shape, nb_classes, pre_model, config)

    model.summary()

    # b = model.layers[1].get_weights()

    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=1500,
                     verbose=True, validation_data=(x_test, y_test), callbacks=callbacks)

    # a = model.layers[1].get_weights()

    # compare_weights(a,b)

    best_model = tf.keras.models.load_model(file_path)

    y_pred2 = best_model.predict(x_test, batch_size=batch_size)

    return_df_metrics = False

    if return_df_metrics:
        y_pred2 = np.argmax(y_pred2, axis=1)
        df_metrics2 = calculate_metrics(y_true, y_pred2, 0.0)
        return_v2 = df_metrics2
    else:
        test_duration2 = time.time() - start_time
        save_test_duration(write_output_dir + 'test_duration2.csv', test_duration2)
        return_v2 = y_pred2

    # save predictions
    np.save(write_output_dir + 'y_pred2.npy', return_v2)

    # convert the predicted from binary to integer
    y_pred2 = np.argmax(return_v2, axis=1)

    df_metrics2 = save_logs(write_output_dir, hist, y_pred2, y_true, test_duration2,
                            plot_test_acc=False)

    tf.keras.backend.clear_session()

    with open(write_output_dir + 'Transfer_learning_run.json', 'a+') as f:
        json.dump({'dataset': dataset_name,
                   'depth': config['depth'],
                   'nb_filters': config['nb_filters'],
                   'batch_size': config['batch_size'],
                   'kernel_size': config['kernel_size'],
                   'use_residual': config['use_residual'],
                   'use_bottleneck': config['use_bottleneck'],
                   'acc': df_metrics2['accuracy'][0],
                   'precision': df_metrics2['precision'][0],
                   'recall': df_metrics2['recall'][0],
                   'budget_ran': 1500,
                   'duration': test_duration2,
                   }, f)
        f.write("\n")

    accu = df_metrics2['accuracy'][0]

    if 0.75 <= accu < 0.9:
        return 0.95 - accu

    if 0.9 <= accu <= 1.0:
        return (1 - accu) / 2

    return accu


def build_model(input_shape, nb_classes, pre_model=None, config=None):
    depth = config['depth'] if config['depth'] else 6
    use_residual = config['use_residual'] if config['use_residual'] else True
    nb_filters = config['nb_filters'] if config['nb_filters'] else 32
    use_bottleneck = config['use_bottleneck'] if config['use_bottleneck'] else True
    kernel_size = config['kernel_size'] if config['kernel_size'] else 41

    input_layer = tf.keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x, use_bottleneck, nb_filters, kernel_size)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    output_layer = tf.keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    if pre_model is not None:

        for i in range(len(model.layers) - 1):
            model.layers[i].set_weights(pre_model.layers[i].get_weights())

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model

