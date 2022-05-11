import math
import pandas as pd
import numpy as np
import os
import json
import torch
import tqdm
import shutil

from smbo import propose_location, expected_improvement
from utils.utils import root_dir
import fsbo
from fsbo import expected_improvement_fsbo

results_dir = root_dir + '/Results/'
# dist_smfo = pd.read_csv(root_dir + '/tsc_transferhpoSMFO_distancing.csv')
dist_fawaz = pd.read_csv(root_dir + '/Distances_csv/datassimilar-datasets_hwaz_m.csv')
# dist_anand = pd.read_csv(root_dir + '/datassimilar-datasets_anand2_m3.csv')
w_dist = pd.read_csv(root_dir + '/Distances_csv/datassimilar-datasets_anand_w_dist1.csv')
# dist_anand_agg2 = pd.read_csv(root_dir + '/Distances_csv/datassimilar-datasets_anand2_m3_aggr2.csv')
# dist_anand_agg3 = pd.read_csv(root_dir + '/Distances_csv/datassimilar-datasets_anand2_m3_aggr3.csv')
dist_anand_agg4 = pd.read_csv(root_dir + '/Distances_csv/datassimilar-datasets_anand_m3_agg4.csv')
# dist_anand_agg5 = pd.read_csv(root_dir + '/Distances_csv/datassimilar-datasets_anand2_m3_aggr5.csv')

folders = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
           'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
           'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
           'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
           'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays',
           # 'ElectricDevices',
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




# ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'Coffee',
#             'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
#             'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
#             'Earthquakes', 'ECG200', 'ECGFiveDays', 'FaceAll', 'FaceFour',
#             'FISH', 'Gun_Point', 'Ham', 'Haptics', 'Herring', 'InlineSkate', 'ItalyPowerDemand',
#             'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'Meat', 'MedicalImages',
#             'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
#             'MoteStrain', 'OliveOil', 'OSULeaf', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
#             'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
#             'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
#             'SonyAIBORobotSurfaceII', 'Strawberry', 'SwedishLeaf', 'Symbols',
#             'synthetic_control', 'ToeSegmentation1', 'Trace', 'TwoLeadECG',
#             'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass']


def choose_method(n_ws, df2, method='m1'):
    if method == 'm3':
        temp1 = 0
        temp2 = math.floor((len(df2) - 1) / n_ws)
        run_through = []
        itt = 0
        while itt < n_ws:
            run_through.append(temp1)
            itt += 1
            temp1 += temp2
        return run_through

    elif method == 'm2':
        temp1 = 0
        temp2 = len(df2) - 1
        run_through = []
        itt = 0
        while itt < n_ws:
            run_through.append(temp1)
            run_through.append(temp2)
            itt += 2
            temp1 += 1
            temp2 -= 1
        return run_through

    elif method == 'm1':
        run_through = [0, 1, 2, 3, 4]
        return run_through


def extract_grid(r_dir):
    df = pd.read_json(r_dir) # , lines=True)
    df['use_bottleneck'] = df['use_bottleneck'].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
    df['use_residual'] = df['use_residual'].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
    df_grid_m = df[['depth', 'nb_filters', 'batch_size', 'kernel_size', 'use_residual', 'use_bottleneck']]
    df_acc_m = df['acc']
    grid_matrix_value = df_grid_m.values
    acc_matrix_value = df_acc_m.values
    return grid_matrix_value, acc_matrix_value


def flatten(t):
    return [item for sublist in t for item in sublist]


def dissolving_grid(datasets):
    work_dir = results_dir + 'trial_benchmark_sorted/'
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)
    for each in datasets:
        # print(each)
        r_d = root_dir + '/Final_benchmarks' + '/Running_' + each + '.json'
        print(r_d)
        df = pd.read_json(r_d)  # , lines=True)
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


def extracting_indices(run_length, df2, grid_m, acc_m, save_dir, iterations_ran=0, method=None):
    X_init = []
    Y_init = []

    run_through = choose_method(run_length, df2, method=method)

    for ran_count in run_through:
        if method == 'greedy':
            ee = int(ran_count)
        else:
            index = np.where((grid_m[:, 4] == df2['config'][ran_count][4]) &  #
                             (grid_m[:, 1] == df2['config'][ran_count][1]) &  #
                             (grid_m[:, 3] == df2['config'][ran_count][3]) &  #
                             (grid_m[:, 5] == df2['config'][ran_count][5]) &  #
                             (grid_m[:, 0] == df2['config'][ran_count][0]) &  #
                             (grid_m[:, 2] == df2['config'][ran_count][2])  #
                             )[0].tolist()
            # 'depth', 'nb_filters', 'batch_size', 'kernel_size', 'use_residual', 'use_bottleneck']]
            # # use_bottleneck nb_filters Batch_size use_residual depth kernel_size --- fsbo
            # ['use_residual', 'use_bottleneck', 'nb_filters', 'batch_size', 'depth', 'kernel_size'] --grid
            ee = int(index[0])
            # print(ee)
        try:
            X = grid_m[ee]
            Y = float(acc_m[ee])
            grid_m = np.delete(grid_m, int(ee), 0)
            acc_m = np.delete(acc_m, int(ee), 0)

            X_init.append(X)
            Y_init.append(Y)

            with open(save_dir, 'a+') as f:
                json.dump({'config': X.tolist(),
                           'accuracy': Y
                           }, f)
                f.write("\n")
            iterations_ran += 1

        except Exception as e:
            print(e)
            raise ValueError('Fix the exception : ', e)

    return X_init, Y_init, grid_m, acc_m, iterations_ran


def fitting_smbo(gpr, w_dir, total_iters_ran, n_iters, grid_matrix, acc_matrixx, X_s, Y_s):
    while total_iters_ran < n_iters:
        gpr.fit(X_s, Y_s)

        X_next, curr_inc = propose_location(expected_improvement, X_s, Y_s, gpr, grid_matrix)

        if X_next is None:
            continue

        Y_next = np.array(acc_matrixx[curr_inc])

        X_val = grid_matrix[int(curr_inc), :].astype(int)
        Y_val = Y_next.reshape(-1, 1)

        with open(w_dir, 'a+') as f:
            json.dump({'config': grid_matrix[curr_inc].tolist(),
                       'accuracy': float(acc_matrixx[curr_inc])
                       }, f)
            f.write("\n")

        X_s = np.vstack((X_s, X_val))
        Y_s = np.vstack((Y_s, Y_val))

        grid_matrix = np.delete(grid_matrix, int(curr_inc), 0)
        acc_matrixx = np.delete(acc_matrixx, int(curr_inc), 0)

        total_iters_ran += 1

    return X_s, Y_s, total_iters_ran


def fsbo_running(n_warm_start, n_iter, total_data, work_dir, fsbo_train_epochs, fsbo_tune_epochs, transfer=False,
                 n_split=5, tf_method='m1', dist_data=None, cc=None, frozen=False, lr_rate=0.0001):
    # copying files to avoid emory conflicts for parallel FSBO runs
    # copying the sorted folder
    to_copy_files1 = os.listdir(results_dir + 'trial_benchmark_sorted/')
    target_dir1 = results_dir + 'trial_benchmark_sorted' + str(cc) + '/'
    if not os.path.exists(target_dir1):
        os.mkdir(target_dir1)
    for files in to_copy_files1:
        shutil.copy(results_dir + 'trial_benchmark_sorted/' + files, target_dir1)

    # copying the original benchmark folder
    to_copy_files2 = os.listdir(root_dir + '/Final_benchmarks')
    target_dir2 = root_dir + '/Final_benchmarks' + str(cc) + '/'
    if not os.path.exists(target_dir2):
        os.mkdir(target_dir2)
    for files in to_copy_files2:
        shutil.copy(root_dir + '/Final_benchmarks/' + files, target_dir2)

    parent_dir = root_dir + '/Final_benchmarks' + str(cc) + '/Running_'

    folders_split = np.array_split(total_data, n_split)

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
            train_data[tr_data]["X"], train_data[tr_data]["y_val"] = retrieve_matrices \
                (parent_dir, tr_data)

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

        model_path = root_dir + '/FSBO_backup' + str(cc) + '/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        conf = {
            "context_size": 10,
            "device": "cpu",
            "lr": lr_rate,
            "model_path": model_path + 'model.pt',
            "use_perf_hist": False,
            "loss_tol": 0.0001,
            "kernel": "rbf",
            "ard": None
        }

        fsbo_func = fsbo.FSBO(train_data, validation_data, conf, feature_extractor)
        print("\n *****************************************************************************************************"
              "\n Training FSBO :")
        fsbo_func.train(epochs=fsbo_train_epochs)

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
                if os.path.exists(work_dir + 'FSBO_' + each_data + '.json'):
                    shutil.rmtree(work_dir)
                    os.mkdir(work_dir)

                ind1 = total_data.index(each_data)
                k1 = dist_data['K_1'][ind1]
                print(each_data, k1)

                readd_dir = target_dir1 + '/RS_' + k1 + '.json'
                df_fsbo = pd.read_json(readd_dir, lines=True)

                run_through = choose_method(n_warm_start, df_fsbo, tf_method)

                for ran_count in run_through:
                    index = np.where((X_MATRIX[:, 0] == df_fsbo['config'][ran_count][0]) &
                                     (X_MATRIX[:, 1] == df_fsbo['config'][ran_count][1]) &
                                     (X_MATRIX[:, 2] == df_fsbo['config'][ran_count][2]) &
                                     (X_MATRIX[:, 3] == df_fsbo['config'][ran_count][3]) &
                                     (X_MATRIX[:, 4] == df_fsbo['config'][ran_count][4]) &
                                     (X_MATRIX[:, 5] == df_fsbo['config'][ran_count][5])
                                     )[0].tolist()
                    try:
                        ee = index[0]
                        print(ee)
                        ind_ref = np.where(np.all(X_MATRIX_ref == X_MATRIX[ee], axis=1))[0].tolist()
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
                        # continue
                        raise ValueError('Clear the exception : ', e)

            else:
                if os.path.exists(work_dir + 'FSBO_' + each_data + '.json'):
                    shutil.rmtree(work_dir)
                    os.mkdir(work_dir)

                for curr_inc in n_warm_start:
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

                fsbo_func.finetuning(x_spt, y_spt, w=None, epochs=fsbo_tune_epochs, freeze=frozen)

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

    shutil.rmtree(model_path)
    shutil.rmtree(target_dir1)
    shutil.rmtree(target_dir2)
