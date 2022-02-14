import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import ConfigSpace
import pickle
import os
import argparse

from ConfigSpace import hyperparameters as CSH
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from smbo import propose_location, expected_improvement
from inception import objective
from transfer_learning import train
from utils.constants import UNIVARIATE_DATASET_NAMES, UNIVARIATE_ARCHIVE_NAMES
from utils.utils import root_dir

parser = argparse.ArgumentParser()


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

results_dir = root_dir + '/Results'


if __name__ == '__main__':
    parser.add_argument('--nargs', nargs='+')

    n_iter = 200
    iterations_ran = 0
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52)

    folders = UNIVARIATE_DATASET_NAMES
    for _, folders in parser.parse_args()._get_kwargs():
        for name in folders:
            print("Dataset: ", name)

            folder_directory = results_dir + '/SMBO/' + name
            if not os.path.exists(folder_directory):
                os.makedirs(folder_directory)

            grid_matrix = create_gridm()

            dim = 6
            X_init = []
            Y_init = []

            for x0 in np.random.randint(low=0, high=grid_matrix.shape[0], size=(10,)):
                temp = grid_matrix[x0, :].astype(int)

                conf = {'depth': int(temp[4]), 'nb_filters': int(temp[2]),
                        'batch_size': int(temp[3]), 'kernel_size': int(temp[5]),
                        'use_residual': bool(temp[0]), 'use_bottleneck': bool(temp[1])}

                tempo = objective(conf, name, run='SMBO',output_dir=folder_directory + '/' + 'config' + str(iterations_ran))
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
                Y_temp = objective(conf, name, run='SMBO', output_dir=folder_directory + '/' + 'config' + str(iterations_ran))
                Y_next = np.array(Y_temp)

                X_val = grid_matrix[int(index), :].astype(int)
                Y_val = Y_next.reshape(-1, 1)

                # Add sample to previous samples
                X_sample = np.vstack((X_sample, X_val))
                Y_sample = np.vstack((Y_sample, Y_val))

                grid_matrix = np.delete(grid_matrix, int(index), 0)
                iterations_ran += 1