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
df_ba = pd.read_csv(r"E:\thesis_work\tsc_transferhpo\datassimilar-datasets_hwaz_m.csv")
n_warm_start = 10
folders = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
           'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
           'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
           'Earthquakes', 'ECG200', 'ECGFiveDays', 'FaceAll', 'FaceFour',
           'FISH', 'Gun_Point', 'Ham', 'Haptics', 'Herring', 'ItalyPowerDemand',
           'Lighting2', 'Lighting7', 'Meat', 'MiddlePhalanxOutlineAgeGroup',
           'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'OliveOil', 'OSULeaf', 'Plane',
           'ProximalPhalanxTW', 'ShapeletSim', 'Wine']


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


def fsbo_running(work_dir, xxx, tr_epochs, ft_epochs, lr, transfer=False, freeze=False):
    n_split = 5
    parent_dir = results_dir + 'kfolds_RS_benchmarks' + str(xxx) + '/Running_'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    n_iter = 100

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


        # print(test_data.keys())

        data_dim = train_data[np.random.choice(list(train_data.keys()), 1).item()]["X"].shape[1]
        feature_extractor = fsbo.MLP(data_dim, n_hidden=32, n_layers=3, n_output=None,
                                     batch_norm=False,
                                     dropout_rate=0.0,
                                     use_cnn=False)

        conf = {
            "context_size": 10,
            "device": "cpu",
            "lr": lr,
            "model_path": 'E:/thesis_work/tsc_transferhpo/Results/trial_benchmark_sorted' + str(xxx) + '/model.pt',
            "use_perf_hist": False,
            "loss_tol": 0.0001,
            "kernel": "rbf",
            "ard": None
        }

        fsbo_func = fsbo.FSBO(train_data, validation_data, conf, feature_extractor)
        print("\n *****************************************************************************************************"
              "\n Training FSBO :")
        fsbo_func.train(epochs=tr_epochs)

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
                readd_dir = results_dir + 'trial_benchmark_sorted' + str(xxx) + '/RS_' + k1 + '.json'
                df_fsbo = pd.read_json(readd_dir, lines=True)
                for ran_count in range(n_warm_start):
                    index = np.where((X_MATRIX[:, 0] == df_fsbo['config'][ran_count][4]) &
                                     (X_MATRIX[:, 1] == df_fsbo['config'][ran_count][2]) &
                                     (X_MATRIX[:, 2] == df_fsbo['config'][ran_count][3]) &
                                     (X_MATRIX[:, 3] == df_fsbo['config'][ran_count][5]) &
                                     (X_MATRIX[:, 4] == df_fsbo['config'][ran_count][0]) &
                                     (X_MATRIX[:, 5] == df_fsbo['config'][ran_count][1])
                                     )[0].tolist()

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

                fsbo_func.finetuning(x_spt, y_spt, w=None, epochs=ft_epochs, freeze=freeze)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help="name and number")
    parser.add_argument("tre", type=int, help="train epochs")
    parser.add_argument("tse", type=int, help="test epochs")
    parser.add_argument("lr", type=float, help="test epochs")
    parser.add_argument("freeze", type=int, help="frozen network?")
    args = parser.parse_args()
    n = args.n
    tr_epochs = args.tre
    ft_epochs = args.tse
    lr = args.lr
    if args.freeze == 1:
        freeze = True
    else:
        freeze = False

    print('CONFIG :', n, tr_epochs, ft_epochs, lr, freeze)

    work_dir = results_dir + 'trial_fsbo' + str(n) + '/'
    fsbo_running(work_dir, n, tr_epochs, ft_epochs, lr, transfer=False, freeze=freeze)