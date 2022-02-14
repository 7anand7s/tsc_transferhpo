import tensorflow as tf
import numpy as np
import sklearn
import os
import time
import json

from utils.utils import read_all_datasets, transform_labels
from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration


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


def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                        padding='same', use_bias=False)(input_tensor)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    x = tf.keras.layers.Add()([shortcut_y, out_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def _inception_module(input_tensor, use_bottleneck, nb_filters, kernel_size, stride=1, bottleneck_size=32,
                      activation='linear'):
    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = tf.keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                 padding='same', activation=activation, use_bias=False)(
            input_tensor)
    else:
        input_inception = input_tensor

    kernel_size = kernel_size - 1
    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                strides=stride, padding='same', activation=activation,
                                                use_bias=False)(
            input_inception))

    max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                    padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = tf.keras.layers.concatenate(conv_list, axis=2)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    return x


root_dir = '/home/fr/fr_fr/fr_aa367/tsc_transferhpo'


def objective(config, dataset_name, run, output_dir=None):
    plot_test_acc = False
    return_df_metrics = False
    budget = 1500
    classifier_name = 'inception'
    archive_name = ARCHIVE_NAMES[0]

    datasets_dict = read_all_datasets(root_dir + '/data', archive_name)



    # for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
    print('\t\t\tdataset_name: ', dataset_name)

    x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data(datasets_dict,
                                                                                           dataset_name)
    if output_dir is None:
        tmp_output_directory = root_dir + '/Results/' + classifier_name + '/' + archive_name + '/'
        output_dir = tmp_output_directory + dataset_name + '/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_directory = output_dir + '/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    nb_epochs = int(budget)

    input_shape = x_train.shape[1:]

    depth = config['depth'] if config['depth'] else 6
    use_residual = config['use_residual'] if config['use_residual'] else True
    nb_filters = config['nb_filters'] if config['nb_filters'] else 32
    use_bottleneck = config['use_bottleneck'] if config['use_bottleneck'] else True
    batch_size = config['batch_size'] if config['batch_size'] else 64
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

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.0001)

    file_path = output_directory + 'best_model.hdf5'

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                          save_best_only=True)

    callbacks = [reduce_lr, model_checkpoint]

    model.summary()

    model.save_weights(output_directory + 'model_init.hdf5')

    if batch_size is None:
        mini_batch_size = int(min(x_train.shape[0] / 10, 16))
    else:
        mini_batch_size = batch_size

    start_time1 = time.time()

    # if plot_test_acc:
    #
    #     hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
    #                      validation_data=(x_test, y_test), verbose=True, callbacks=callbacks)
    # else:

    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs, verbose=True,
                     callbacks=callbacks)

    duration1 = time.time() - start_time1

    model.save(output_directory + 'last_model.hdf5')

    best_model = tf.keras.models.load_model(file_path)

    start_time2 = time.time()

    y_pred2 = best_model.predict(x_test, batch_size=batch_size)
    if return_df_metrics:
        y_pred2 = np.argmax(y_pred2, axis=1)
        df_metrics2 = calculate_metrics(y_true, y_pred2, 0.0)
        return_v2 = df_metrics2
    else:
        test_duration2 = time.time() - start_time2
        save_test_duration(output_directory + 'test_duration2.csv', test_duration2)
        return_v2 = y_pred2

    # save predictions
    np.save(output_directory + 'y_pred2.npy', return_v2)

    # convert the predicted from binary to integer
    y_pred2 = np.argmax(return_v2, axis=1)

    df_metrics2 = save_logs(output_directory, hist, y_pred2, y_true, duration1,
                            plot_test_acc=plot_test_acc)

    tf.keras.backend.clear_session()

    if type(config['use_residual']) == bool:
        pass
    else:
        config['use_residual'] = bool(config['use_residual'])

    if run:
        with open(root_dir + '/Results/SMBO/' + '_run_%s%s.json' % (run, dataset_name), 'a+') as f:
            json.dump({'dataset': dataset_name,
                       'depth': config['depth'],
                       'nb_filters': config['nb_filters'],
                       'batch_size': config['batch_size'],
                       'kernel_size': config['kernel_size'],
                       'use_residual': 'True' if config['use_residual'] else 'False',
                       'use_bottleneck': 'True' if config['use_bottleneck'] else 'False',
                       'acc': df_metrics2['accuracy'][0],
                       'precision': df_metrics2['precision'][0],
                       'recall': df_metrics2['recall'][0],
                       'budget_ran': budget,
                       'duration': duration1,
                       }, f)
            f.write("\n")

    accu = df_metrics2['accuracy'][0]

    if 0.75 <= accu < 0.9:
        return 0.95 - accu

    if 0.9 <= accu <= 1.0:
        return (1 - accu) / 2

    return accu
