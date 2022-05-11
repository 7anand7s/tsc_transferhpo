import time

import tensorflow as tf
import numpy as np
import os
import json

from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.utils import calculate_metrics
from utils.utils import root_dir
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class CustomStopper(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0.0, patience=0,
                 verbose=0, mode='auto', start_epoch=100):  # add argument for starting epoch
        super(CustomStopper, self).__init__(monitor=monitor, patience=patience, min_delta=min_delta,
                                            mode=mode, verbose=verbose)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


def extract_data(dataset_name, split_n=5):
    file_name = root_dir + '/data/UCR_TS_Archive_2015/TSC/' + dataset_name + '/' + dataset_name
    data1 = np.loadtxt(file_name + '_TRAIN', delimiter=',')
    data2 = np.loadtxt(file_name + '_TEST', delimiter=',')
    tot_data = shuffle(np.concatenate((data1, data2), axis=0))
    nb_classes = len(np.unique(tot_data[:, 0]))
    Y_data = tot_data[:, 0]
    X_data = tot_data[:, 1:]
    encoder = LabelEncoder()
    encoder.fit(Y_data)
    Y_data = encoder.transform(Y_data)

    enc = OneHotEncoder()
    enc.fit(Y_data.reshape(-1, 1))
    y_ehc_data = enc.transform(Y_data.reshape(-1, 1)).toarray()
    split_xdata = np.array_split(X_data, split_n)
    split_ytdata = np.array_split(Y_data, split_n)
    split_ydata = np.array_split(y_ehc_data, split_n)

    return nb_classes, X_data, Y_data, split_xdata, split_ydata, split_ytdata


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


def objective_renewed(config, dataset_name, run, n_splits=5, output_dir=None):
    d = time.time()
    budget = 1500
    classifier_name = 'inception'
    archive_name = ARCHIVE_NAMES[0]

    # for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
    print('\t\t\tdataset_name: ', dataset_name)

    nb_classes, X_data, Y_data, split_xdata, split_ydata, split_ytdata = extract_data(dataset_name, 5)

    i = 0
    train_curve = []
    val_curve = []
    test_acc = []
    test_pres = []
    test_recall = []

    while i < n_splits:
        if i == n_splits:
            break

        # for count in range(n_splits):
        d1 = split_xdata[i:n_splits] + split_xdata[-n_splits:i - n_splits]
        d2 = split_ydata[i:n_splits] + split_ydata[-n_splits:i - n_splits]
        d3 = split_ytdata[i:n_splits] + split_ytdata[-n_splits:i - n_splits]
        k = 0
        x_val = d1[k]
        y_val = d2[k]
        x_test = d1[k + 1]
        y_test = d2[k + 1]
        y_test_true = d3[k + 1]
        x_train = np.vstack(d1[k + 2:])
        y_train = np.vstack(d2[k + 2:])
        i += 1

        if len(x_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        # y_train_ehc, y_val_ehc, y_test_ehc, ytest_true = brush_data(y_train, y_val, y_test)

        if output_dir is None:
            tmp_output_directory = root_dir + '/Results/' + classifier_name + '/' + archive_name + '/'
            output_dir = tmp_output_directory + dataset_name

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

        save_dir = root_dir + '/Results/inception/TSC/' + dataset_name

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_path = save_dir + '/best_model.hdf5'

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                              save_best_only=True)

        early_stopping = CustomStopper(monitor='val_loss', start_epoch=500, patience=20, min_delta=0.001, mode='min')

        callbacks = [reduce_lr, model_checkpoint, early_stopping]

        model.summary()

        model.save_weights(save_dir + 'model_init.hdf5')

        if batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = batch_size

        hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs, verbose=True,
                         validation_data=(x_val, y_val), callbacks=callbacks)

        val_curve.append(np.array(hist.history['val_loss']))
        train_curve.append(np.array(hist.history['loss']))

        model.save(save_dir + 'last_model.hdf5')

        best_model = tf.keras.models.load_model(file_path)

        y_pred2 = best_model.predict(x_test, batch_size=batch_size)

        y_pred2 = np.argmax(y_pred2, axis=1)

        print(y_test_true)
        print(y_pred2)

        df_metrics = calculate_metrics(y_test_true, y_pred2, 0.0)

        print(df_metrics)

        tf.keras.backend.clear_session()

        accu = df_metrics['accuracy'][0]
        precs = df_metrics['precision'][0]
        recall = df_metrics['recall'][0]

        print(accu)
        test_acc.append(accu)
        test_pres.append(precs)
        test_recall.append(recall)

    min_length_tr = min(map(len, train_curve))
    for each in range(len(train_curve)):
        train_curve[each] = train_curve[each][:min_length_tr]

    min_length_tst = min(map(len, val_curve))
    for each in range(len(val_curve)):
        val_curve[each] = val_curve[each][:min_length_tst]

    train_cur = np.average(train_curve, axis=0)
    val_cur = np.average(val_curve, axis=0)

    acc = sum(test_acc) / len(test_acc)
    precision = sum(test_pres) / len(test_pres)
    recal = sum(test_recall) / len(test_recall)

    if type(config['use_residual']) == bool:
        pass
    else:
        config['use_residual'] = bool(config['use_residual'])

    # os.remove(output_directory + 'last_model.hdf5')
    # os.remove(output_directory + 'best_model.hdf5')
    # os.remove(output_directory + 'model_init.hdf5')
    ran_for = time.time() - d
    if run:
        with open(root_dir + '/Results/%s/' % run + 'Running_%s.json' % dataset_name, 'a+') as f:
            json.dump({'dataset': dataset_name,
                       'depth': config['depth'],
                       'nb_filters': config['nb_filters'],
                       'batch_size': config['batch_size'],
                       'kernel_size': config['kernel_size'],
                       'use_residual': 'True' if config['use_residual'] else 'False',
                       'use_bottleneck': 'True' if config['use_bottleneck'] else 'False',
                       'acc': acc,
                       'precision': precision,
                       'recall': recal,
                       'budget_ran': len(val_cur),
                       'train_curve': train_cur.tolist(),
                       'val_curve': val_cur.tolist(),
                       'train_curve 1': train_curve[0].tolist(),
                       'val_curve 1': val_curve[0].tolist(),
                       'acc 1': test_acc[0],
                       'precision 1': test_pres[0],
                       'recall 1': test_recall[0],
                       'train_curve 2': train_curve[1].tolist(),
                       'val_curve 2': val_curve[1].tolist(),
                       'acc 2': test_acc[1],
                       'precision 2': test_pres[1],
                       'recall 2': test_recall[1],
                       'train_curve 3': train_curve[2].tolist(),
                       'val_curve 3': val_curve[2].tolist(),
                       'acc 3': test_acc[2],
                       'precision 3': test_pres[2],
                       'recall 3': test_recall[2],
                       'train_curve 4': train_curve[3].tolist(),
                       'val_curve 4': val_curve[3].tolist(),
                       'acc 4': test_acc[3],
                       'precision 4': test_pres[3],
                       'recall 4': test_recall[3],
                       'train_curve 5': train_curve[4].tolist(),
                       'val_curve 5': val_curve[4].tolist(),
                       'acc 5': test_acc[4],
                       'precision 5': test_pres[4],
                       'recall 5': test_recall[4],
                       }, f)
            f.write("\n")

    if 0.75 <= acc < 0.9:
        return 0.95 - acc

    if 0.9 <= acc <= 1.0:
        return (1 - acc) / 2

    return acc
