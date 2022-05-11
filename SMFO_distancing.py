import time
import pandas as pd
import json
import os
import numpy as np

from utils.utils import root_dir


def extract_acc(lambda_value, df_matrix):
    indexx = np.where((lambda_value[0] == df_matrix[:, 0]) &
                      (lambda_value[1] == df_matrix[:, 1]) &
                      (lambda_value[2] == df_matrix[:, 2]) &
                      (lambda_value[3] == df_matrix[:, 3]) &
                      (lambda_value[4] == df_matrix[:, 4]) &
                      (lambda_value[5] == df_matrix[:, 5]))
    acc = df_matrix[indexx[0][0], 6]
    return acc


datasets = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
           'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
           'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
           'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
           'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays',
           'ElectricDevices',
           'FaceAll', 'FaceFour',
           'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines',
           'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
           'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages',
           'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
           'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil',
           'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
           'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
           'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
           'SonyAIBORobotSurfaceII', 'Strawberry', 'SwedishLeaf', 'Symbols',
           'synthetic_control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
           'Two_Patterns', 'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
           'uWaveGestureLibrary_Z', 'wafer', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga']




def SMFO_dist():
    run_folder = r'E:/thesis_work/tsc_transferhpo/Results/kfolds_RS_benchmarks'
    if os.path.exists(run_folder + '/RS_sampled_configs.json'):
        with open(run_folder + '/RS_sampled_configs.json', "r") as jsonfile:
            conf_list = json.load(jsonfile)
    list_conf = conf_list["configs"]

    d = []
    for i, conf in enumerate(list_conf):
        d.append([conf['depth'], conf['nb_filters'], conf['batch_size'], conf['kernel_size'],
                  conf['use_residual'], conf['use_bottleneck']])

    df_ref = pd.DataFrame(d, columns=['depth', 'nb_filters', 'batch_size', 'kernel_size', 'use_residual',
                                      'use_bottleneck'])
    df_ref_matrix = df_ref.to_numpy()

    nb_neighbors = len(datasets) - 1
    columns = [('K_' + str(i)) for i in range(1, nb_neighbors + 1)]
    neighbors = pd.DataFrame(data=np.zeros((len(datasets), nb_neighbors),
                                           dtype=np.str_), columns=columns, index=datasets)

    for data_one in datasets:
        df1 = pd.read_json( root_dir + '/Final_benchmarks/Running_' + data_one + '.json')
                # , lines=True)

        df1['use_bottleneck'] = df1['use_bottleneck'].map({'True': True, 'False': False, 'true': True, 'false': False})
        df1['use_residual'] = df1['use_residual'].map({'True': True, 'False': False, 'true': True, 'false': False})
        df1 = df1.drop(
            columns=['dataset', 'precision', 'recall', 'budget_ran', 'train_curve', 'val_curve', 'train_curve 1',
                     'val_curve 1', 'acc 1', 'precision 1', 'recall 1', 'train_curve 2',
                     'val_curve 2', 'acc 2', 'precision 2', 'recall 2', 'train_curve 3',
                     'val_curve 3', 'acc 3', 'precision 3', 'recall 3', 'train_curve 4',
                     'val_curve 4', 'acc 4', 'precision 4', 'recall 4', 'train_curve 5',
                     'val_curve 5', 'acc 5', 'precision 5', 'recall 5'])
        df1_matrix = df1.to_numpy()
        distance_dict = {}
        for data_two in datasets:
            df2 = pd.read_json(
                root_dir + '/Final_benchmarks/Running_' + data_two + '.json') #,
                # lines=True)

            df2['use_bottleneck'] = df2['use_bottleneck'].map(
                {'True': True, 'False': False, 'true': True, 'false': False})
            df2['use_residual'] = df2['use_residual'].map({'True': True, 'False': False, 'true': True, 'false': False})
            df2 = df2.drop(
                columns=['dataset', 'precision', 'recall', 'budget_ran', 'train_curve', 'val_curve', 'train_curve 1',
                         'val_curve 1', 'acc 1', 'precision 1', 'recall 1', 'train_curve 2',
                         'val_curve 2', 'acc 2', 'precision 2', 'recall 2', 'train_curve 3',
                         'val_curve 3', 'acc 3', 'precision 3', 'recall 3', 'train_curve 4',
                         'val_curve 4', 'acc 4', 'precision 4', 'recall 4', 'train_curve 5',
                         'val_curve 5', 'acc 5', 'precision 5', 'recall 5'])
            df2_matrix = df2.to_numpy()

            # Do not proceed when D1==D2
            if data_one == data_two:
                continue
            #
            dist = 0
            print(data_one, data_two)
            for lamda_1 in df_ref_matrix:
                for lamda_2 in df_ref_matrix:
                    # print(lamda_1, lamda_2)
                    # F(lamda_1, D1)
                    acc_l1_d1 = extract_acc(lamda_1, df1_matrix)
                    # indexx_a1 = np.where(lamda_1.all() == df1_matrix[:, :6].all())[0][0]
                    # acc_a1 = df1_matrix[indexx_a1, 6]
                    # F(lamda_2, D1)
                    acc_l2_d1 = extract_acc(lamda_2, df1_matrix)
                    # F(lamda_1, D2)
                    acc_l1_d2 = extract_acc(lamda_1, df2_matrix)
                    # F(lamda_2, D2)
                    acc_l2_d2 = extract_acc(lamda_2, df2_matrix)

                    a = (acc_l1_d1 > acc_l2_d1)
                    b = (acc_l1_d2 > acc_l2_d2)
                    if a ^ b:
                        dist += 1
                    # print(a, b)
                    # raise ValueError('A very specific bad thing happened.')
                    # print('done')

            distance_dict[data_two] = dist
        dist_dict_sorted = dict(sorted(distance_dict.items(), key=lambda item: item[1]))
        print(dist_dict_sorted)
        i = 1
        for k in dist_dict_sorted.keys():
            print(i, k)
            neighbors.loc[data_one]['K_' + str(i)] = str(k)
            i += 1
    neighbors.to_csv(root_dir + 'SMFO_distancing_84.csv')

    # raise ValueError('A very specific bad thing happened.')


def retrieve_matrices(directory, dataset_name):
    df = pd.read_json(directory + dataset_name + '.json') # , lines=True)

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


def flatten(t):
    return [item for sublist in t for item in sublist]


def greedy_iter():
    folders = datasets
    n_split = 5
    n_warm_start = 5
    parent_dir = root_dir + '/Final_benchmarks/Running_'
    # work_dir = root_dir + '/Results/trial_fsbo/'
    # if not os.path.exists(work_dir):
    #     os.mkdir(work_dir)
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

        Acc_add_temp = np.zeros([200, 1])
        opt_hp = []
        for tr_data in train_folders:
            Grid_train, Acc_train = retrieve_matrices(parent_dir, tr_data)
            # print(Acc_add_temp.shape, Acc_train.shape)
            Acc_add_temp = np.hstack((Acc_add_temp, Acc_train.reshape([200, 1])))

        summation_acc = np.sum(Acc_add_temp, axis=1)
        print(summation_acc.shape)
        print(np.max(summation_acc), np.argmax(summation_acc))
        opt_hp.append(np.argmax(summation_acc))

        while len(opt_hp) < n_warm_start:
            Acc_add_temp = np.zeros([200 - len(opt_hp), 1])
            for tr_data in train_folders:
                Grid_train, Acc_train = retrieve_matrices(parent_dir, tr_data)
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
            print(summation_acc.shape)
            print(np.max(summation_acc), np.argmax(summation_acc))
            opt_hp.append(np.argmax(summation_acc))

        print('############ yes #######################')
        print(opt_hp)
        print(val_folders.flatten(), test_folders.flatten())

        # for v_data in val_folders.flatten():
        #     Grid_test1, Acc_test1 = retrieve_matrices(parent_dir, v_data)
        #
        # for ts_data in test_folders.flatten():
        #     Grid_test2, Acc_test2 = retrieve_matrices(parent_dir, ts_data)


# greedy_iter()

# print(time.time())
SMFO_dist()