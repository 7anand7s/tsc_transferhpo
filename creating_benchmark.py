import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import ConfigSpace
import pickle
import os
import argparse
import json

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

    run_name = 'benchmark'
    run_folder = results_dir + '/' + run_name

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    parser.add_argument('--nargs', nargs='+')

    n_iter = 200

    cs = ConfigSpace.ConfigurationSpace()

    depth = CSH.CategoricalHyperparameter('depth', [6, 3, 9])
    use_residual = CSH.CategoricalHyperparameter('use_residual', [True, False])
    nb_filters = CSH.CategoricalHyperparameter('nb_filters', [32, 16, 64])
    use_bottleneck = CSH.CategoricalHyperparameter('use_bottleneck', [True, False])
    batch_size = CSH.CategoricalHyperparameter('batch_size', [64, 16, 32, 128])
    kernel_size = CSH.CategoricalHyperparameter('kernel_size', [41, 32, 64])

    cs.add_hyperparameters([depth, use_residual, nb_filters, use_bottleneck, batch_size, kernel_size])

    if not os.path.exists(run_folder + '/RS_sampled_configs.json'):
        print("nothing")
        configurations = cs.sample_configuration(n_iter)
        with open(run_folder + '/RS_sampled_configs.json', 'a+') as f:
            conf_list = {'configs': [i.get_dictionary() for i in configurations]}
            print(conf_list)
            json.dump(conf_list, f)
            f.write("\n")
    else:
        with open(run_folder + "/RS_sampled_configs.json", "r") as jsonfile:
            conf_list = json.load(jsonfile)

    folders = UNIVARIATE_DATASET_NAMES
    for _, folders in parser.parse_args()._get_kwargs():
        for name in folders:

            folder_directory = run_folder + '/' + name
            if not os.path.exists(folder_directory):
                os.makedirs(folder_directory)

            iteration_number = 0

            print(name)
            iteration_to_run=0
            if os.path.exists(run_folder + '/' + name + '/Running_' + name + '.json'):
                df = pd.read_json(run_folder + '/' + name + '/Running_' + name + '.json', lines=True)
                iteration_to_run = int(df.shape[0]) + 1

            print("Running iteration: ", iteration_number)

            # dataset_names:
            for conf in list(conf_list.values())[0]:
                if iteration_number < iteration_to_run:
                    iteration_number += 1
                    continue
                objective(conf, name, run='benchmark',output_dir=folder_directory + '/' + 'config' + str(iteration_number))
                iteration_number += 1