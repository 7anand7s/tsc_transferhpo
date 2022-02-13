import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import ConfigSpace
import pickle
import os

from ConfigSpace import hyperparameters as CSH
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from smbo import propose_location, expected_improvement
from inception import objective
from transfer_learning import train
from utils.constants import UNIVARIATE_DATASET_NAMES, UNIVARIATE_ARCHIVE_NAMES
from utils.utils import read_all_datasets


def get_conf2(ind, grid_matrixx):
    temp2 = grid_matrixx[ind, :].astype(int)

    conf2 = {'depth': int(temp2[4]), 'nb_filters': int(temp2[2]),
             'batch_size': int(temp2[3]), 'kernel_size': int(temp2[5]),
             'use_residual': bool(temp2[0]), 'use_bottleneck': bool(temp2[1])}
    return conf2


def extr_values(df, max_index):
    conf = {}
    conf['depth'] = int(df['depth'].loc[max_index])
    conf['nb_filters'] = int(df['nb_filters'].loc[max_index])
    conf['batch_size'] = int(df['batch_size'].loc[max_index])
    conf['kernel_size'] = int(df['kernel_size'].loc[max_index])
    conf['use_residual'] = df['use_residual'].loc[max_index]
    conf['use_bottleneck'] = df['use_bottleneck'].loc[max_index]
    return conf


def create_gridm():
    m1 = [True, False]
    m2 = [True, False]
    m3 = [16, 32, 64]
    m4 = [16, 32, 64, 128]
    m5 = [3, 6, 9]
    m6 = [32, 41, 64]
    n_size = len(m1) * len(m2) * len(m3) * len(m4) * len(m5) * len(m6)
    gridm_matrix = np.zeros([n_size, 6])
    counter = 0
    for useResidual in m1:
        for useBottleneck in m2:
            for nbFilters in m3:
                for batchSize in m4:
                    for dept in m5:
                        for kernelSize in m6:
                            gridm_matrix[counter, :] = [useResidual, useBottleneck, nbFilters, batchSize, dept,
                                                        kernelSize]
                            counter += 1
    return gridm_matrix


root_dir = '/home/fr/fr_fr/fr_aa367/tsc_transferhpo'
results_dir = root_dir + '/Results'

if sys.argv[1] == 'hpo_smbo':
    n_iter = 50
    iterations_ran = 0
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52)

    folders = UNIVARIATE_DATASET_NAMES
    for name in folders:
        grid_matrix = create_gridm()

        dim = 6
        X_init = []
        Y_init = []

        for x0 in np.random.randint(low=0, high=grid_matrix.shape[0], size=(5,)):
            temp = grid_matrix[x0, :].astype(int)

            conf = {'depth': int(temp[4]), 'nb_filters': int(temp[2]),
                    'batch_size': int(temp[3]), 'kernel_size': int(temp[5]),
                    'use_residual': bool(temp[0]), 'use_bottleneck': bool(temp[1])}

            tempo = objective(conf, name, run='SMBO_run_')
            X_init.append(temp)
            Y_init.append(tempo)

            grid_matrix = np.delete(grid_matrix, x0, 0)
            iterations_ran += 1

        # Initialize samples
        X_sample = np.array(X_init).reshape(-1, dim)
        Y_sample = np.array(Y_init).reshape(-1, 1)

        while iterations_ran < n_iter:
            # Update Gaussian process with existing samples
            gpr.fit(X_sample, Y_sample)

            indices = np.arange(0, grid_matrix.shape[0])

            # Obtain next sampling point from the acquisition function (expected_improvement)
            X_next, index = propose_location(expected_improvement, X_sample, Y_sample, gpr, grid_matrix)

            if X_next is None:
                continue

            # objective
            conf = get_conf2(int(index), grid_matrix)
            Y_temp = objective(conf, name, run='SMBO_run_')
            Y_next = np.array(Y_temp)

            X_val = grid_matrix[int(index), :].astype(int)
            Y_val = Y_next.reshape(-1, 1)

            # Add sample to previous samples
            X_sample = np.vstack((X_sample, X_val))
            Y_sample = np.vstack((Y_sample, Y_val))

            grid_matrix = np.delete(grid_matrix, int(index), 0)
            iterations_ran += 1

if sys.argv[1] == 'hpo_rs':

    folders = UNIVARIATE_DATASET_NAMES

    cs = ConfigSpace.ConfigurationSpace()

    depth = CSH.CategoricalHyperparameter('depth', [6, 3, 9])
    use_residual = CSH.CategoricalHyperparameter('use_residual', [True, False])
    nb_filters = CSH.CategoricalHyperparameter('nb_filters', [32, 16, 64])
    use_bottleneck = CSH.CategoricalHyperparameter('use_bottleneck', [True, False])
    batch_size = CSH.CategoricalHyperparameter('batch_size', [64, 16, 32, 128])
    kernel_size = CSH.CategoricalHyperparameter('kernel_size', [41, 32, 64])

    cs.add_hyperparameters([depth, use_residual, nb_filters, use_bottleneck, batch_size, kernel_size])

    for name in folders:
        configurations = cs.sample_configuration(50)
        # configurations.append(cs.get_default_configuration())
        for i in configurations:
            try:
                objective(config=i.get_dictionary(), dataset_name=name, run='RS_run_')
            except:
                continue

if sys.argv[1] == 'create_bestmodels':
    for count, names in enumerate(UNIVARIATE_DATASET_NAMES):

        filee_path3 = results_dir + '/SMBO/smbo_gp_' + names + '.json'
        columns = ["dataset", "depth", "nb_filters", "batch_size", "kernel_size", "use_residual", "use_bottleneck",
                   "acc", "precision", "recall", "budget_ran", "duration"]

        df = pd.read_csv(filee_path3, names=columns)

        # reading and trimming data from csv to pandas dataframe format
        df['dataset'] = df['dataset'].str.slice(start=12)
        df['depth'] = df['depth'].str.slice(start=9).astype(int)
        df['acc'] = df['acc'].str.slice(start=7).astype(float)
        df['batch_size'] = df['batch_size'].str.slice(start=14).astype(int)
        df['kernel_size'] = df['kernel_size'].str.slice(start=15).astype(int)
        df['use_residual'] = df['use_residual'].str.slice(start=16).map({' false': False, ' true': True})
        df['use_bottleneck'] = df['use_bottleneck'].str.slice(start=19).map({'false': False, 'true': True})
        df['nb_filters'] = df['nb_filters'].str.slice(start=14).astype(int)
        df['precision'] = df['precision'].str.slice(start=13).astype(float)
        df['recall'] = df['recall'].str.slice(start=10).astype(float)
        df['budget_ran'] = df['budget_ran'].str.slice(start=15).astype(int)
        df['duration'] = df['duration'].str.slice(start=12, stop=-1).astype(float)

        best = 1
        # Find the best configurations
        for each in df['acc'].nlargest(5).index:

            # max_index = df["acc"].idxmax()
            best_model_f = results_dir + '/SMBO/BestModel_' + names + '/'
            if not os.path.exists(best_model_f):
                os.mkdir(best_model_f)

            configuration = extr_values(df, each)

            c_file = open(best_model_f + 'bestconfig' + str(best) + '.pkl', "wb")
            pickle.dump(configuration, c_file)
            c_file.close()
            if best == 1:
                pass
                # objective(configuration, dataset_name=names, run=None, output_dir=best_model_f)
            best += 1

write_dir_root = root_dir + '/transfer-learning-results'

if sys.argv[1] == 'transfer_learning':
    if not os.path.exists(write_dir_root):
        os.mkdir(write_dir_root)
    datasets_dict = read_all_datasets(root_dir + '/data', UNIVARIATE_ARCHIVE_NAMES[0])
    # loop through all datasets
    for dataset_name in UNIVARIATE_DATASET_NAMES:

        # # load the model to transfer to other datasets
        # pre_model = tf.keras.models.load_model(output_dir + 'best_model.hdf5')
        #
        # # output file path for the new tranfered re-trained model
        # file_path = write_dir_root + '/' + dataset_name + '/' + dataset_name_tranfer + 'best_model.hdf5'
        #
        # # callbacks : reduce learning rate
        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        #
        # # model checkpoint
        # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
        #                                                       save_best_only=True)
        # callbacks = [reduce_lr, model_checkpoint]
        #
        # get the directory of the model for this current dataset_name#
        out_dir = write_dir_root + '/' + dataset_name
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        pd_df = pd.read_csv('/home/fr/fr_fr/fr_aa367/tsc_transferhpo/Results/datassimilar-datasets_hwaz_m.csv')
        k_nn = np.where(pd_df['dataset'] == dataset_name, pd_df['K_1'], 0)
        dataset_name_tranfer = k_nn[k_nn != 0][0]
        
        output_dir = results_dir + '/SMBO/BestModel_' + dataset_name_tranfer + '/'

        # for dataset_name_tranfer in UNIVARIATE_DATASET_NAMES:
        # iterations to run for Gaussian Process
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gpr = GaussianProcessRegressor(kernel=m52)
        iter_ran = 0
        no_iter = 50

        print('Tranfering from ' + dataset_name + ' to ' + dataset_name_tranfer)

        top5_configs = []
        for counting in range(1, 6):
            b_file = open(output_dir + "bestconfig" + str(counting) + ".pkl", "rb")
            confi = pickle.load(b_file)
            top5_configs.append(confi)

        gridMatrix = create_gridm()
        dim = 6
        Xt_init = []
        Yt_init = []
        
        for confi in top5_configs:

            temp_l = np.array([confi['use_residual'], confi['use_bottleneck'], confi['nb_filters'],
                               confi['batch_size'], confi['depth'], confi['kernel_size']])
            # temp_l = temp_l.reshape(-1, dim)
            for i, matrix in enumerate(gridMatrix):
                cc = 0
                for j, k in zip(matrix, temp_l):
                    cc = cc + 1 if j == k else cc
                if cc == 6:
                    index = i

            tempo = objective(confi, dataset_name=dataset_name, run='Transfer_learning_run_',
                                  output_dir=write_dir_root + '/' + dataset_name + '/')
            Yt_init.append(tempo)

            Xt_init.append(gridMatrix[index, :].astype(int))

            gridMatrix = np.delete(gridMatrix, int(index), 0)

        # Initialize samples
        Xt_sample = np.array(Xt_init).reshape(-1, dim)
        Yt_sample = np.array(Yt_init).reshape(-1, 1)

        while iter_ran < no_iter:

            # Update Gaussian process with existing samples
            gpr.fit(Xt_sample, Yt_sample)

            indices = np.arange(0, gridMatrix.shape[0])

            # Obtain next sampling point from the acquisition function (expected_improvement)
            X_next, index = propose_location(expected_improvement, Xt_sample, Yt_sample, gpr, gridMatrix)

            if X_next is None:
                continue

            # objective
            confi = get_conf2(int(index), gridMatrix)
            Y_temp = objective(confi, dataset_name=dataset_name, run='Transfer_learning_run_',
                               output_dir=write_dir_root + '/' + dataset_name + '/')
            Y_next = np.array(Y_temp)

            X_val = gridMatrix[int(index), :].astype(int)
            Y_val = Y_next.reshape(-1, 1)

            # Add sample to previous samples
            X_sample = np.vstack((Xt_sample, X_val))
            Y_sample = np.vstack((Yt_sample, Y_val))

            gridMatrix = np.delete(gridMatrix, int(index), 0)
            iter_ran += 1

if sys.argv[1] == 'plot_graphs':
    pass
