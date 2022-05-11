import math
import sys
import pandas as pd
import numpy as np
import os
import json
import torch
import tqdm
import argparse

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from smbo import propose_location, expected_improvement
from utils.utils import root_dir
from utils.constants import UNIVARIATE_DATASET_NAMES
import fsbo
from fsbo import expected_improvement_fsbo

results_dir = root_dir + '/Results/'
df_ba = pd.read_csv(r"E:\thesis_work\tsc_transferhpo\datassimilar-datasets_anand2_m3.csv")
n_warm_start = 5
folders = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'Coffee',
           'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
           'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
           'Earthquakes', 'ECG200', 'ECGFiveDays', 'FaceAll', 'FaceFour',
           'FISH', 'Gun_Point', 'Ham', 'Haptics', 'Herring', 'InlineSkate', 'ItalyPowerDemand',
           'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'Meat', 'MedicalImages',
           'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
           'MoteStrain', 'OliveOil', 'OSULeaf', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
           'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
           'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
           'SonyAIBORobotSurfaceII', 'Strawberry', 'SwedishLeaf', 'Symbols',
           'synthetic_control', 'ToeSegmentation1', 'Trace', 'TwoLeadECG',
           'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass']


# 62


# Ham 200, Earthquakes 199, Haptics 200, ItalyPowerDemand 174

def flatten(t):
    return [item for sublist in t for item in sublist]


def retrieve_matrices(directory, dataset_name):
    df = pd.read_json(directory + dataset_name + '.json', lines=True)

    df['depth'] = df['depth'].astype(int)
    df['nb_filters'] = df['nb_filters'].astype(int)
    df['kernel_size'] = df['kernel_size'].astype(int)
    df['batch_size'] = df['batch_size'].astype(int)
    df['use_bottleneck'] = df['use_bottleneck'].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
    df['use_residual'] = df['use_residual'].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
    df['use_residual'] = df['use_residual'].astype(int)
    df['use_bottleneck'] = df['use_bottleneck'].astype(int)
    acc_matrix = df['acc'].to_numpy()
    df = df.drop(
        columns=['dataset', 'acc', 'precision', 'recall', 'budget_ran', 'train_curve', 'val_curve', 'train_curve 1',
                 'val_curve 1', 'acc 1', 'precision 1', 'recall 1', 'train_curve 2',
                 'val_curve 2', 'acc 2', 'precision 2', 'recall 2', 'train_curve 3',
                 'val_curve 3', 'acc 3', 'precision 3', 'recall 3', 'train_curve 4',
                 'val_curve 4', 'acc 4', 'precision 4', 'recall 4', 'train_curve 5',
                 'val_curve 5', 'acc 5', 'precision 5', 'recall 5'])

    grid_matrix = df.to_numpy()
    return grid_matrix, acc_matrix


def retrieve_matrices2(directory, dataset_name):
    df = pd.read_json(directory + dataset_name + '.json', lines=True)

    df['depth'] = df['depth'].astype(int)
    df['nb_filters'] = df['nb_filters'].astype(int)
    df['kernel_size'] = df['kernel_size'].astype(int)
    df['batch_size'] = df['batch_size'].astype(int)
    df['use_bottleneck'] = df['use_bottleneck'].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
    df['use_residual'] = df['use_residual'].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
    df['use_residual'] = df['use_residual'].astype(int)
    df['use_bottleneck'] = df['use_bottleneck'].astype(int)
    # acc_matrix = df['acc'].values
    # df = df.drop(
    #     columns=['dataset', 'acc', 'precision', 'recall', 'budget_ran', 'train_curve', 'val_curve', 'train_curve 1',
    #              'val_curve 1', 'acc 1', 'precision 1', 'recall 1', 'train_curve 2',
    #              'val_curve 2', 'acc 2', 'precision 2', 'recall 2', 'train_curve 3',
    #              'val_curve 3', 'acc 3', 'precision 3', 'recall 3', 'train_curve 4',
    #              'val_curve 4', 'acc 4', 'precision 4', 'recall 4', 'train_curve 5',
    #              'val_curve 5', 'acc 5', 'precision 5', 'recall 5'])
    #
    # grid_matrix = df.values
    grid_matrix = df[['depth', 'nb_filters', 'batch_size', 'kernel_size', 'use_residual', 'use_bottleneck']].values
    acc_matrix = df['acc'].values
    return grid_matrix, acc_matrix


def fitting_smbo(w_dir, iterations_ran, n_iter, grid_m, acc_m, X_s, Y_s):
    while iterations_ran < n_iter:
        gpr.fit(X_s, Y_s)

        X_next, curr_inc = propose_location(expected_improvement, X_s, Y_s, gpr, grid_m)

        if X_next is None:
            continue

        Y_next = np.array(acc_m[curr_inc])

        X_val = grid_m[int(curr_inc), :].astype(int)
        Y_val = Y_next.reshape(-1, 1)

        with open(w_dir, 'a+') as f:
            json.dump({'config': grid_m[curr_inc].tolist(),
                       'accuracy': float(acc_m[curr_inc])
                       }, f)
            f.write("\n")

        X_s = np.vstack((X_s, X_val))
        Y_s = np.vstack((Y_s, Y_val))

        grid_m = np.delete(grid_m, int(curr_inc), 0)
        acc_m = np.delete(acc_m, int(curr_inc), 0)

        iterations_ran += 1

    return X_s, Y_s, iterations_ran


def random_start_smbo(n, grid_matrix, acc_matrix, iterations_ran, save_dir):
    X_init = []
    Y_init = []
    for _ in range(n):
        # for curr_inc in n:
        d_size = grid_matrix.shape[0] - 1
        curr_inc = np.random.randint(0, d_size)

        X_init.append(grid_matrix[curr_inc].astype(int))
        Y_init.append(float(acc_matrix[curr_inc]))

        with open(save_dir, 'a+') as f:
            json.dump({'config': grid_matrix[curr_inc].tolist(),
                       'accuracy': float(acc_matrix[curr_inc])
                       }, f)
            f.write("\n")

        grid_matrix = np.delete(grid_matrix, int(curr_inc), 0)
        acc_matrix = np.delete(acc_matrix, int(curr_inc), 0)

        iterations_ran += 1

    return grid_matrix, acc_matrix, iterations_ran, X_init, Y_init


def extract_grid(r_dir):
    df = pd.read_json(r_dir, lines=True)
    df['use_bottleneck'] = df['use_bottleneck'].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
    df['use_residual'] = df['use_residual'].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
    df_grid_m = df[['depth', 'nb_filters', 'batch_size', 'kernel_size', 'use_residual', 'use_bottleneck']]
    df_acc_m = df['acc']
    grid_m = df_grid_m.values
    acc_m = df_acc_m.values
    return grid_m, acc_m


def extracting_indices(run_length, df2, grid_m, acc_m, iterations_ran=None, save_dir=None, write_mode=False,
                       run_matrix=False, n_ws=None, method=None):
    X_init = []
    Y_init = []

    if run_matrix:
        run_through = run_length
    else:
        run_through = range(run_length - 1)

    if method == 'm3':
        temp1 = 0
        temp2 = math.floor((len(df2) - 1) / n_ws)
        run_through = []
        itt = iterations_ran
        while itt < n_ws:
            run_through.append(temp1)
            # run_through.append(temp2)
            itt += 1
            temp1 += temp2
            # temp2 -= 1

    if method == 'm2':
        temp1 = 0
        temp2 = len(df2) - 1
        # temp2 = math.floor((len(df2) - 1) / n_ws)
        run_through = []
        itt = iterations_ran
        while itt < n_ws:
            run_through.append(temp1)
            run_through.append(temp2)
            itt += 2
            temp1 += 1
            temp2 -= 1

    if method == 'm1':
        run_through = [0, 1, 2, 3, 4]

    print(run_through)
    for ran_count in run_through:
        if method == 'greedy':
            ee = int(ran_count)
            print(ee)
        else:
            print(ran_count, df2['config'][ran_count])
            print(grid_m)
            index = np.where((grid_m[:, 4] == df2['config'][ran_count][4]) &  #
                             (grid_m[:, 1] == df2['config'][ran_count][1]) &  #
                             (grid_m[:, 3] == df2['config'][ran_count][3]) &  #
                             (grid_m[:, 5] == df2['config'][ran_count][5]) &  #
                             (grid_m[:, 0] == df2['config'][ran_count][0]) &  #
                             (grid_m[:, 2] == df2['config'][ran_count][2])  #
                             )[0].tolist()

            # # use_bottleneck nb_filters Batch_size use_residual depth kernel_size --- fsbo
            # ['use_residual', 'use_bottleneck', 'nb_filters', 'batch_size', 'depth', 'kernel_size'] --grid

            # index = np.where((grid_m[:, 4] == df2['config'][ran_count][0]) &  #
            #                  (grid_m[:, 2] == df2['config'][ran_count][1]) &  #
            #                  (grid_m[:, 3] == df2['config'][ran_count][2]) &  #
            #                  (grid_m[:, 5] == df2['config'][ran_count][3]) &  #
            #                  (grid_m[:, 0] == df2['config'][ran_count][4]) &  #
            #                  (grid_m[:, 1] == df2['config'][ran_count][5])  #
            #                  )[0].tolist()
            # print(index)
            ee = index[0]
            # with open('saved_c.txt', 'a+') as f:
            #     f.write('%d,' %ee)
        try:
            X = grid_m[ee]
            Y = float(acc_m[ee])
            grid_m = np.delete(grid_m, int(ee), 0)
            acc_m = np.delete(acc_m, int(ee), 0)

            X_init.append(X)
            Y_init.append(Y)

            if write_mode:
                with open(save_dir, 'a+') as f:
                    json.dump({'config': X.tolist(),
                               'accuracy': Y
                               }, f)
                    f.write("\n")
                iterations_ran += 1

                if iterations_ran > n_ws:
                    break

        except Exception as e:
            print(e)
            continue
    if write_mode:
        return X_init, Y_init, grid_m2, acc_m2, iterations_ran
    return X_init, Y_init


if sys.argv[1] == 'dissolve_grid':
    work_dir = results_dir + 'trial_benchmark_sorted/'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    for each in folders:
        r_d = results_dir + 'kfolds_RS_benchmarks/Running_' + each + '.json'
        df = pd.read_json(r_d, lines=True)
        df_s = df.sort_values(by="acc", ascending=False)
        acc_sorted_index = df_s.index.tolist()
        g1, a1 = extract_grid(r_d)
        # print(acc_sorted_index)
        for curr_inc in acc_sorted_index:
            with open(work_dir + 'RS_' + each + '.json', 'a+') as f:
                json.dump({'config': g1[curr_inc].tolist(),
                           'accuracy': float(a1[curr_inc])
                           }, f)
                f.write("\n")

if sys.argv[1] == 'hpo_rs':
    n_iter = 50
    run_folder = results_dir + 'kfolds_RS_benchmarks/'
    work_dir = results_dir + 'trial_RS/'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    for name in folders:
        try:
            iterations_ran = 0

            read_dir = run_folder + '/Running_' + name + '.json'
            grid_m, acc_m = extract_grid(read_dir)

            if os.path.exists(work_dir + 'RS_' + name + '.json'):
                df_rs = pd.read_json(work_dir + 'RS_' + name + '.json', lines=True)
                iterations_ran = int(df_rs.shape[0])

            while iterations_ran < n_iter:
                data_size = grid_m.shape[0] - 1
                curr_inc = np.random.randint(0, data_size)

                with open(work_dir + 'RS_' + name + '.json', 'a+') as f:
                    json.dump({'config': grid_m[curr_inc].tolist(),
                               'accuracy': float(acc_m[curr_inc])
                               }, f)
                    f.write("\n")

                grid_m = np.delete(grid_m, int(curr_inc), 0)
                acc_m = np.delete(acc_m, int(curr_inc), 0)

                iterations_ran += 1
        except Exception as e:
            print(e)
            continue

if sys.argv[1] == 'hpo_smbo':
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52)
    n_iter = 50
    run_folder = results_dir + 'kfolds_RS_benchmarks/'
    smbo_dir = results_dir + 'trial_SMBO/'
    if not os.path.exists(smbo_dir):
        os.mkdir(smbo_dir)

    with open('data_txt.txt') as f:
        for line in f:
            line = line.split()  # to deal with blank
            if line:  # lines (ie skip them)
                line = [int(i) for i in line]

    for name in folders:
        this_rand_init = []
        for _ in range(n_warm_start):
            this_rand_init.append(line.pop(0))
        # folders:
        try:
            iterations_ran = 0

            read_dir = run_folder + '/Running_' + name + '.json'
            grid_m, acc_m = extract_grid(read_dir)
            dim = 6
            X_init = []
            Y_init = []

            if os.path.exists(smbo_dir + 'SMBO_' + name + '.json'):
                df_smbo = pd.read_json(smbo_dir + 'SMBO_' + name + '.json', lines=True)
                iterations_ran = int(df_smbo.shape[0])

                if iterations_ran > 0:
                    X_init, Y_init = extracting_indices(iterations_ran, df_smbo, grid_m, acc_m)

            else:
                saving_dir = smbo_dir + 'SMBO_' + name + '.json'
                grid_m, acc_m, iterations_ran, X_init, Y_init \
                    = random_start_smbo(n_warm_start, grid_m, acc_m, iterations_ran, saving_dir)

            # Initialize samples
            X_sample = np.array(X_init).reshape(-1, dim)
            Y_sample = np.array(Y_init).reshape(-1, 1)

            use_dir = smbo_dir + 'SMBO_' + name + '.json'
            X_sample, Y_sample, iterations_ran = fitting_smbo(use_dir, iterations_ran, n_iter, grid_m, acc_m, X_sample,
                                                              Y_sample)

        except Exception as e:
            print(e)
            continue

write_dir_root = root_dir + '/transfer-learning-results'

if sys.argv[1] == 'count':
    for name in UNIVARIATE_DATASET_NAMES:
        try:
            df = pd.read_json('E:/thesis_work/tsc_transferhpo/Results/kfolds_RS_benchmarks/Running_' + name + '.json',
                              lines=True)
            print(name, len(df))
        except Exception as e:
            print(name, e)
            pass

if sys.argv[1] == 'tflr_m1':
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52)
    work_dir = results_dir + 'trial_tflr_smbo_a2/'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    n_iter = 50

    for loop1, ind_c in zip(folders, range(len(df_ba))):
        iterations_ran = 0
        k1 = df_ba['K_1'][ind_c]
        k2 = df_ba['K_2'][ind_c]
        k3 = df_ba['K_3'][ind_c]
        k4 = df_ba['K_4'][ind_c]
        k5 = df_ba['K_5'][ind_c]

        for loop2 in [k1]:
            dim = 6
            read_dir = results_dir + 'kfolds_RS_benchmarks' + '/Running_' + loop1 + '.json'
            grid_m2, acc_m2 = extract_grid(read_dir)
            print(loop1, loop2)

            X_init = []
            Y_init = []

            if os.path.exists(results_dir + 'trial_benchmark_sorted' + '/RS_' + loop2 + '.json'):
                df3 = pd.read_json(results_dir + 'trial_benchmark_sorted' + '/RS_' + loop2 + '.json', lines=True)
                warm_started_bool = True

                # print(acc_sorted_index)

                if warm_started_bool:
                    savin_dir = work_dir + loop1 + '.json'
                    X_init, Y_init, grid_m2, acc_m2, iterations_ran \
                        = extracting_indices(range(n_warm_start), df3, grid_m2, acc_m2,
                                             iterations_ran, savin_dir, write_mode=True,
                                             run_matrix=True, n_ws=n_warm_start)
                #   print(X_init, Y_init, grid_m2, acc_m2, iterations_ran)

            else:
                print("cannot find warm start configs")

        # Initialize samples
        X_sample = np.array(X_init).reshape(-1, dim)
        Y_sample = np.array(Y_init).reshape(-1, 1)

        use_dir = work_dir + loop1 + '.json'
        X_sample, Y_sample, iterations_ran = fitting_smbo(use_dir, iterations_ran, n_iter, grid_m2, acc_m2, X_sample,
                                                          Y_sample)

if sys.argv[1] == 'tflr_greedy_start':
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52)
    work_dir = results_dir + 'GP_greedy_start/'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    n_iter = 50
    n_split = 5
    parent_dir = root_dir + '/Results/kfolds_RS_benchmarks/Running_'

    folders_split = np.array_split(folders, n_split)

    i = 0
    while i < n_split:
        if i == n_split:
            break
        d1 = folders_split[i:n_split] + folders_split[-n_split:i - n_split]
        k = 0
        test_folders = d1[k]
        val_folders = d1[k + 1]
        train_folders = flatten(d1[k + 2:])
        i += 1

        Acc_add_temp = np.zeros([200, 1])
        opt_hp = []
        for tr_data in train_folders:
            Grid_train, Acc_train = retrieve_matrices2(parent_dir, tr_data)
            Acc_add_temp = np.hstack((Acc_add_temp, Acc_train.reshape([200, 1])))

        summation_acc = np.sum(Acc_add_temp, axis=1)
        opt_hp.append(np.argmax(summation_acc))

        while len(opt_hp) < n_warm_start:
            Acc_add_temp = np.zeros([200 - len(opt_hp), 1])
            for tr_data in train_folders:
                Grid_train, Acc_train = retrieve_matrices2(parent_dir, tr_data)
                b_m = []
                for each_ind in opt_hp:
                    b_m.append(Acc_train[each_ind])
                    Grid_train = np.delete(Grid_train, int(each_ind), 0)
                    Acc_train = np.delete(Acc_train, int(each_ind), 0)

                b = max(b_m)
                for indexing in range(len(Acc_train)):
                    if b > Acc_train[indexing]:
                        Acc_train[indexing] = b
                # print(Acc_add_temp.shape, Acc_train.shape)
                Acc_add_temp = np.hstack((Acc_add_temp, Acc_train.reshape([len(Acc_train), 1])))

            summation_acc = np.sum(Acc_add_temp, axis=1)
            opt_hp.append(np.argmax(summation_acc))

        for v_data in val_folders.flatten():
            iterations_ran = 0
            dim = 6
            val_file_path = results_dir + 'kfolds_RS_benchmarks' + '/Running_' + v_data + '.json'
            df3 = pd.read_json(val_file_path, lines=True)
            grid_m2, acc_m2 = retrieve_matrices2(parent_dir, v_data)
            savin_dir = work_dir + v_data + '.json'
            X_init, Y_init, grid_m2, acc_m2, iterations_ran \
                = extracting_indices(opt_hp, df3, grid_m2, acc_m2,
                                     iterations_ran, savin_dir, write_mode=True,
                                     run_matrix=True, n_ws=n_warm_start, method='greedy')

            # Initialize samples
            X_sample = np.array(X_init).reshape(-1, dim)
            Y_sample = np.array(Y_init).reshape(-1, 1)

            use_dir = work_dir + v_data + '.json'
            X_sample, Y_sample, iterations_ran = fitting_smbo(use_dir, iterations_ran, n_iter, grid_m2, acc_m2,
                                                              X_sample, Y_sample)

        for ts_data in test_folders.flatten():
            iterations_ran = 0
            dim = 6
            val_file_path = results_dir + 'kfolds_RS_benchmarks' + '/Running_' + ts_data + '.json'
            df3 = pd.read_json(val_file_path, lines=True)
            grid_m2, acc_m2 = retrieve_matrices2(parent_dir, ts_data)
            savin_dir = work_dir + ts_data + '.json'
            X_init, Y_init, grid_m2, acc_m2, iterations_ran \
                = extracting_indices(opt_hp, df3, grid_m2, acc_m2,
                                     iterations_ran, savin_dir, write_mode=True,
                                     run_matrix=True, n_ws=n_warm_start, method='greedy')

            # Initialize samples
            X_sample = np.array(X_init).reshape(-1, dim)
            Y_sample = np.array(Y_init).reshape(-1, 1)

            use_dir = work_dir + ts_data + '.json'
            X_sample, Y_sample, iterations_ran = fitting_smbo(use_dir, iterations_ran, n_iter, grid_m2, acc_m2,
                                                              X_sample, Y_sample)

# method2 --> warm start with both top-lowest and top-highest configurations
# method3 --> warm start with all sorted values -100configs? 1st,10th,20th,30th, so on

if sys.argv[1] == 'tflr_m2':
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52)
    work_dir = results_dir + 'trial_SMBO/'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    n_iter = 50

    for loop1, ind_c in zip(folders, range(len(df_ba))):
        iterations_ran = 0
        k1 = df_ba['K_1'][ind_c]
        k2 = df_ba['K_2'][ind_c]
        k3 = df_ba['K_3'][ind_c]
        k4 = df_ba['K_4'][ind_c]
        k5 = df_ba['K_5'][ind_c]

        for loop2 in [k1]:
            dim = 6
            read_dir = results_dir + 'kfolds_RS_benchmarks' + '/Running_' + loop1 + '.json'
            grid_m2, acc_m2 = extract_grid(read_dir)
            print(loop1, loop2)

            X_init = []
            Y_init = []

            if os.path.exists(results_dir + 'trial_benchmark_sorted' + '/RS_' + loop2 + '.json'):
                df3 = pd.read_json(results_dir + 'trial_benchmark_sorted' + '/RS_' + loop2 + '.json', lines=True)
                warm_started_bool = True
                # if os.path.exists(results_dir + 'trial_fsbo' + '/FSBO_' + loop1 + '.json'):
                #     df3 = pd.read_json(results_dir + 'trial_fsbo' + '/FSBO_' + loop1 + '.json', lines=True)
                #     warm_started_bool = True

                # print(acc_sorted_index)

                if warm_started_bool:
                    savin_dir = work_dir + loop1 + '.json'
                    X_init, Y_init, grid_m2, acc_m2, iterations_ran \
                        = extracting_indices(range(n_warm_start), df3, grid_m2, acc_m2,
                                             iterations_ran, savin_dir, write_mode=True,
                                             run_matrix=True, n_ws=n_warm_start, method='m1')
                #   print(X_init, Y_init, grid_m2, acc_m2, iterations_ran)

            else:
                print("cannot find warm start configs")

        # Initialize samples
        X_sample = np.array(X_init).reshape(-1, dim)
        Y_sample = np.array(Y_init).reshape(-1, 1)

        use_dir = work_dir + loop1 + '.json'
        X_sample, Y_sample, iterations_ran = fitting_smbo(use_dir, iterations_ran, n_iter, grid_m2, acc_m2, X_sample,
                                                          Y_sample)

if sys.argv[1] == 'tflr_m3':
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52)
    work_dir = results_dir + 'trial_tflr_new1/'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    n_iter = 50

    for loop1, ind_c in zip(folders, range(len(df_ba))):
        k1 = df_ba['K_1'][ind_c]
        k2 = df_ba['K_2'][ind_c]
        k3 = df_ba['K_3'][ind_c]

        X_SAMPLE = []
        Y_SAMPLE = []
        for loop2 in [k1]:

            iterations_ran = 0

            r_dir = results_dir + 'kfolds_RS_benchmarks' + '/Running_' + loop2 + '.json'
            grid_m, acc_m = extract_grid(r_dir)
            dim = 6

            if os.path.exists(results_dir + 'trial_SMBO/' + 'SMBO_' + loop2 + '.json'):
                df2 = pd.read_json(results_dir + 'trial_SMBO/' + 'SMBO_' + loop2 + '.json', lines=True)
                iterations_ran = int(df2.shape[0])

                if iterations_ran > 0:
                    X_init, Y_init = extracting_indices(iterations_ran, df2, grid_m, acc_m)

                    # Initialize samples
                    X_sample = np.array(X_init).reshape(-1, dim)
                    Y_sample = np.array(Y_init).reshape(-1, 1)

                    X_SAMPLE.append(X_sample)
                    Y_SAMPLE.append(Y_sample)

            else:
                raise ValueError('Run GP-SMBO to use transfer learning')

        X_SAMPLE = np.vstack(X_SAMPLE)
        Y_SAMPLE = np.vstack(Y_SAMPLE)

        r_dir2 = results_dir + 'kfolds_RS_benchmarks' + '/Running_' + loop1 + '.json'
        grid_m2, acc_m2 = extract_grid(r_dir2)

        use_dir = work_dir + loop1 + '.json'
        X_sample, Y_sample, _ = fitting_smbo(use_dir, 0, n_iter, grid_m2, acc_m2, X_SAMPLE, Y_SAMPLE)


def fsbo_running(work_dir, transfer=False):
    n_split = 5
    parent_dir = results_dir + 'kfolds_RS_benchmarks/Running_'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    n_iter = 50

    folders_split = np.array_split(folders, n_split)

    i = 0
    while i < n_split:
        if i == n_split:
            break
        d1 = folders_split[i:n_split] + folders_split[-n_split:i - n_split]
        k = 0
        test_folders = d1[k]
        val_folders = d1[k + 1]
        train_folders = flatten(d1[k + 2:])
        i += 1

        train_data = {}
        validation_data = {}
        test_data = {}
        for tr_data in train_folders:
            train_data[tr_data] = {}
            train_data[tr_data]["X"], train_data[tr_data]["y_val"] = retrieve_matrices(parent_dir, tr_data)

        for v_data in val_folders.flatten():
            validation_data[v_data] = {}
            validation_data[v_data]["X"], validation_data[v_data]["y_val"] = retrieve_matrices(parent_dir, v_data)

        for ts_data in test_folders.flatten():
            test_data[ts_data] = {}
            test_data[ts_data]["X"], test_data[ts_data]["y_val"] = retrieve_matrices(parent_dir, ts_data)

        data_dim = train_data[np.random.choice(list(train_data.keys()), 1).item()]["X"].shape[1]
        feature_extractor = fsbo.MLP(data_dim, n_hidden=32, n_layers=3, n_output=None,
                                     batch_norm=False,
                                     dropout_rate=0.0,
                                     use_cnn=False)

        conf = {
            "context_size": 10,
            "device": "cpu",
            "lr": 0.0001,
            "model_path": 'E:/thesis_work/tsc_transferhpo/benchmark/model.pt',
            "use_perf_hist": False,
            "loss_tol": 0.0001,
            "kernel": "rbf",
            "ard": None
        }

        fsbo_func = fsbo.FSBO(train_data, validation_data, conf, feature_extractor)
        print("\n *****************************************************************************************************"
              "\n Training FSBO :")
        fsbo_func.train(epochs=10000)

        for each_data in test_folders.flatten():

            iter_ran = 0
            status_bar = tqdm.tqdm(total=n_iter)
            task = test_data[each_data]
            X_spt = []
            X_MATRIX = task["X"]
            X_MATRIX_ref = task["X"]
            Y_MATRIX = task["y_val"]
            Y_MATRIX_ref = task["y_val"]

            print("\n ------------------------------------------------------------------------------------------------"
                  "\n Running the test Dataset :", each_data)

            if transfer:
                ind1 = folders.index(each_data)
                k1 = df_ba['K_1'][ind1]
                # k2 = df_ba['K_2'][ind_c]
                # k3 = df_ba['K_3'][ind_c]
                readd_dir = results_dir + 'trial_benchmark_sorted/RS_' + k1 + '.json'
                df_fsbo = pd.read_json(readd_dir, lines=True)
                method = True
                if method:
                    temp1 = 0
                    temp2 = math.floor((len(df_fsbo) - 1) / n_warm_start)
                    run_through = []
                    itt = 0
                    while itt < n_warm_start:
                        run_through.append(temp1)
                        # run_through.append(temp2)
                        itt += 1
                        temp1 += temp2
                        # temp2 -= 1

                print(run_through)
                for ran_count in run_through:

                    # for ran_count in range(n_warm_start):
                    index = np.where((X_MATRIX[:, 0] == df_fsbo['config'][ran_count][4]) &
                                     (X_MATRIX[:, 1] == df_fsbo['config'][ran_count][2]) &
                                     (X_MATRIX[:, 2] == df_fsbo['config'][ran_count][3]) &
                                     (X_MATRIX[:, 3] == df_fsbo['config'][ran_count][5]) &
                                     (X_MATRIX[:, 4] == df_fsbo['config'][ran_count][0]) &
                                     (X_MATRIX[:, 5] == df_fsbo['config'][ran_count][1])
                                     )[0].tolist()
                    # saved as

                    try:
                        ee = index[0]
                        # print(ee)
                        ind_ref = np.where(np.all(X_MATRIX_ref == X_MATRIX[ee], axis=1))[0].tolist()
                        # print(X_MATRIX_ref[ind_ref[0]].tolist())
                        # print(ind_ref)
                        with open(work_dir + 'FSBO_' + each_data + '.json', 'a+') as f:
                            json.dump({'config': X_MATRIX_ref[ind_ref[0]].tolist(),
                                       'accuracy': float(Y_MATRIX_ref[ind_ref[0]])
                                       }, f)
                            f.write("\n")

                        X_MATRIX = np.delete(X_MATRIX, int(ee), 0)
                        Y_MATRIX = np.delete(Y_MATRIX, int(ee), 0)
                        X_spt.append(ind_ref[0])
                        iter_ran += 1
                        if iter_ran > n_warm_start:
                            break
                    except Exception as e:
                        print(e)
                        continue

                # print(X_spt)

            else:
                for _ in range(n_warm_start):
                    data_size = X_MATRIX.shape[0] - 1
                    curr_inc = np.random.randint(0, data_size)

                    ind_ref = np.where(np.all(X_MATRIX_ref == X_MATRIX[curr_inc], axis=1))[0]

                    with open(work_dir + 'FSBO_' + each_data + '.json', 'a+') as f:
                        json.dump({'config': X_MATRIX[curr_inc].tolist(),
                                   'accuracy': float(Y_MATRIX[curr_inc])
                                   }, f)
                        f.write("\n")

                    X_MATRIX = np.delete(X_MATRIX, int(curr_inc), 0)
                    Y_MATRIX = np.delete(Y_MATRIX, int(curr_inc), 0)

                    X_spt.append(ind_ref[0])

                    iter_ran += 1
                    status_bar.update()

            while iter_ran < n_iter:
                x_spt = torch.FloatTensor(task["X"])[X_spt].to("cpu")
                y_spt = torch.FloatTensor(task["y_val"])[X_spt].to("cpu")

                fsbo_func.finetuning(x_spt, y_spt, w=None, epochs=5000, freeze=False)

                search_space = torch.FloatTensor(X_MATRIX).to("cpu")

                X_next, curr_inc = fsbo.propose_location(expected_improvement_fsbo, x_spt, y_spt, fsbo_func,
                                                         search_space)

                ind_ref = np.where(np.all(X_MATRIX_ref == X_MATRIX[curr_inc], axis=1))[0]
                X_spt.append(ind_ref[0])

                with open(work_dir + 'FSBO_' + each_data + '.json', 'a+') as f:
                    json.dump({'config': X_MATRIX[curr_inc].tolist(),
                               'accuracy': float(Y_MATRIX[curr_inc])
                               }, f)
                    f.write("\n")

                X_MATRIX = np.delete(X_MATRIX, int(curr_inc), 0)
                Y_MATRIX = np.delete(Y_MATRIX, int(curr_inc), 0)

                iter_ran += 1
                status_bar.update()


if sys.argv[1] == 'hpo_fsbo':
    work_dir = results_dir + 'trial_fsbo/'
    fsbo_running(work_dir, transfer=False)

if sys.argv[1] == 'hpo_fsbo_tf':
    work_dir = results_dir + 'trial_tflr_fsbo_method3_a2/'
    fsbo_running(work_dir, transfer=True)

if sys.argv[1] == 'plot_graphs':
    np.seterr(divide='ignore', invalid='ignore')

    names = ['rs',
             'GP_without_TL',
             'GP_smfo',
             'GP_Dist_anand',
             'GP_Dist_fawaz',
             # 'FSBO_diverse'
             # 'GP_smfo', #'fsbo',
             # 'GP_anand',
             # 'GP_fawaz', #'smbo_m2_a2', 'smbo_m3_a2',
             # 'GP_greedy', #'smbo_m2_fw', 'smbo_m3_fw',
             # 'fsbo_a2', 'fsbo_m2_a2',  'fsbo_m3_a2',
             # 'fsbo_fw',  'fsbo_m2_fw', 'fsbo_m3_fw',
             ]
    Big_Y = []
    for hehe in range(len(names)):
        Big_Y.append([])
    #  'fsbo', 'fsbo-tf',
    ranks_y = []
    # df_ba = pd.read_csv(r"E:\thesis_work\tsc_transferhpo\datassimilar-datasets_hwaz_m.csv")

    for name, i_l in zip(folders, range(len(df_ba))):
        regrets_y = []
        k1 = df_ba['K_1'][i_l]
        k2 = df_ba['K_2'][i_l]
        k3 = df_ba['K_3'][i_l]

        # try:
        print(name)
        plt.clf()

        dfs = []
        dfs.append(pd.read_json(results_dir + 'trial_RS/RS_' + name + '.json', lines=True))
        dfs.append(pd.read_json(results_dir + 'trial_SMBO/SMBO_' + name + '.json', lines=True))
        # dfs.append(pd.read_json(results_dir + 'trial_fsbo/FSBO_' + name + '.json', lines=True))

        dfs.append(pd.read_json(results_dir + 'trial_tflr_smbo_smfo/' + name + '.json', lines=True))
        # transfer learning with SMBO baseline
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_smbo_a2/' + name + '.json', lines=True))
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_smbo_fw/' + name + '.json', lines=True))
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_smbo_method2_a2/' + name + '.json', lines=True))
        dfs.append(pd.read_json(results_dir + 'trial_tflr_smbo_method3_a2/' + name + '.json', lines=True))
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_smbo_gredy_start/' + name + '.json', lines=True))

        # dfs.append(pd.read_json(results_dir + 'trial_tflr_smbo_fw/' + name + '.json', lines=True))
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_smbo_method2_fw/' + name + '.json', lines=True))
        dfs.append(pd.read_json(results_dir + 'trial_tflr_smbo_method3_fw/' + name + '.json', lines=True))
        # transfer learning with FSBO baseline
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_fsbo_a2/FSBO_' + name + '.json', lines=True))
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_fsbo_method2_a2/FSBO_' + name + '.json', lines=True))
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_fsbo_method3_a2/FSBO_' + name + '.json', lines=True))
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_fsbo_fw/FSBO_' + name + '.json', lines=True))
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_fsbo_method2_fw/FSBO_' + name + '.json', lines=True))
        # dfs.append(pd.read_json(results_dir + 'trial_tflr_fsbo_method3_fw/FSBO_' + name + '.json', lines=True))

        dfs_orig = pd.read_json(results_dir + 'kfolds_RS_benchmarks/Running_' + name + '.json', lines=True)
        # eeeee = os.listdir(results_dir + 'trial_tflr_m2/' + name + '/')
        # for each_tf in eeeee:
        #     dfs.append(pd.read_json(results_dir + 'trial_tflr_m2/' + name + '/' + each_tf, lines=True))
        #     names.append(each_tf)

        dfs_acc = []
        for df in dfs:
            df = df.head(50)
            # print(df.shape)
            dfs_acc.append(df['accuracy'].values if 'accuracy' in df else df['acc'].values)
            # print(dfs_rs)
        # print(dfs_acc)

        data = np.array(dfs_acc)
        # norm_arr = (data - np.min(data)) / (np.max(data) - np.min(data))
        # min_max_scaler = preprocessing.MinMaxScaler()
        # norm_arr = min_max_scaler.fit_transform(data)
        # print(norm_arr)

        # x = []
        # for acc_df in dfs_acc:
        #     xval = ((acc_df - np.min(acc_df)) / (np.max(acc_df) - np.min(acc_df)))
        #     xval[np.isnan(xval)] = 1
        #     x.append(xval)
        #     # print(xval)

        y = []
        for each_x in data:
            y1 = np.zeros(each_x.shape)
            y1[0] = each_x[0]
            for li in range(1, len(each_x)):
                y1[li] = max(each_x[:li])

            y.append(y1)

        # print(y)

        # maximumm = np.max(dfs_orig['accuracy'].values if 'accuracy' in dfs_orig else dfs_orig['acc'].values)
        # print(maximumm)
        # max_pos = []
        # for each in y:
        #     c = 0
        #     for ee in each:
        #         if ee == maximumm:
        #             break
        #         else:
        #             c += 1
        #     max_pos.append(c)
        #
        # max_cur = np.where(np.min(max_pos) == max_pos)[0][0]
        # plt.plot(np.arange(len(y[max_cur])), y[max_cur], label=names[max_cur])

        orig_acc = dfs_orig['accuracy'].values if 'accuracy' in dfs_orig else dfs_orig['acc'].values
        maximum_val = max(orig_acc)
        # print(maximum_val)
        # print(maximum_val, min(y[-1]))
        y_star = np.ones(len(y[0])) * maximum_val
        for iter_y in y:
            regrets_y.append(y_star - iter_y)

        # print(regrets_y)

        rank_df = pd.DataFrame(regrets_y)
        # print(rank_df)

        rankings = rank_df.rank(axis=0)
        # print(rankings)

        for plot_idx in range(len(regrets_y)):
            # print(rankings.iloc[plot_idx])
            plt.plot(np.arange(len(rankings.iloc[plot_idx])), rankings.iloc[plot_idx], label=names[plot_idx])
            Big_Y[plot_idx].append(rankings.iloc[plot_idx])

        # for each in Big_Y:
        #     print(len(each[0]))

        # plt.plot(np.arange(len(rankings.iloc[0])), rankings.iloc[0], 'gold', label=names[0])
        # plt.plot(np.arange(len(rankings.iloc[1])), rankings.iloc[1], 'lime', label=names[1])
        # plt.plot(np.arange(len(rankings.iloc[2])), rankings.iloc[2], 'black', label=names[2])
        # plt.plot(np.arange(len(rankings.iloc[3])), rankings.iloc[3], 'r', label=names[3])
        # plt.plot(np.arange(len(rankings.iloc[4])), rankings.iloc[4], 'blue', label=names[4])
        # plt.plot(np.arange(len(rankings.iloc[5])), rankings.iloc[5], 'purple', label=names[5])
        # plt.plot(np.arange(len(rankings.iloc[6])), rankings.iloc[6], 'green', label=names[6])
        # plt.plot(np.arange(len(rankings.iloc[7])), rankings.iloc[7], 'cyan', label=names[7])
        plt.legend()
        plt.savefig(results_dir + 'trial_Graphs/' + name + '.png')
        #
        # Y_rs.append(rankings.iloc[0])
        # Y_smbo.append(rankings.iloc[1])
        # Y_tf3.append(rankings.iloc[2])
        # Y_tf2.append(rankings.iloc[3])
        # Y_tf1.append(rankings.iloc[4])
        # Y_fsbo.append(rankings.iloc[5])
        # Y_temp1.append(rankings.iloc[6])
        # Y_temp2.append(rankings.iloc[7])

        # for iter_idx in range(len(regrets_y)):
        #     Big_Y[iter_idx].append(rankings.iloc[iter_idx])
    # except:
    #     continue
    # Y1 = sum(Y_rs) / len(Y_rs)
    # Y2 = sum(Y_smbo) / len(Y_smbo)
    # Y3 = sum(Y_tf3) / len(Y_tf3)
    # Y4 = sum(Y_tf2) / len(Y_tf2)
    # Y5 = sum(Y_tf1) / len(Y_tf1)
    # Y6 = sum(Y_fsbo) / len(Y_fsbo)
    # Y7 = sum(Y_temp1) / len(Y_temp1)
    # Y8 = sum(Y_temp2) / len(Y_temp2)
    plt.clf()
    # plt.plot(np.arange(len(Y1)), Y1, 'gold', label=names[0])
    # plt.plot(np.arange(len(Y2)), Y2, 'lime', label=names[1])
    # plt.plot(np.arange(len(Y3)), Y3, 'black', label=names[2])
    # plt.plot(np.arange(len(Y4)), Y4, 'r', label=names[3])
    # plt.plot(np.arange(len(Y5)), Y5, 'blue', label=names[4])
    # plt.plot(np.arange(len(Y6)), Y6, 'purple', label=names[5])
    # plt.plot(np.arange(len(Y7)), Y7, 'green', label=names[6])
    # plt.plot(np.arange(len(Y8)), Y8, 'cyan', label=names[7])
    for iter_index in range(len(Big_Y)):
        temp_v1 = sum(Big_Y[iter_index])
        tempv2 = len(Big_Y[iter_index])
        # print(iter_index, temp_v1, tempv2)
        temp_v = temp_v1 / tempv2
        plt.plot(np.arange(len(temp_v)), temp_v, label=names[iter_index])

    # plt.legend(loc='upper left')
    plt.legend()
    plt.savefig(results_dir + 'trial_Graphs/00_AVERAGED_k2.png')
