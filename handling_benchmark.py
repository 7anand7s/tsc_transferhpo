import pandas as pd
import json
import os
import numpy as np

from k_folds import objective_renewed
from utils.utils import root_dir
from all_functions import folders

datasets = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
            'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
            'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
            'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
            'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
            'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines',
            'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
            'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages',
            'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
            'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil',
            'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
            'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
            'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
            'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
            'synthetic_control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
            'Two_Patterns', 'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
            'uWaveGestureLibrary_Z', 'wafer', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga']


def count():
    for dataset in datasets:
        try:
            df = pd.read_json(
                'E:/thesis_work/tsc_transferhpo/Results/kfolds_RS_benchmarks/Running_' + dataset + '.json',
                lines=True)
            print(dataset, len(df))
        except Exception as e:
            print(dataset, e)
            pass


def check_for_duplicates():
    for dataset in datasets:
        try:
            df = pd.read_json(
                'E:/thesis_work/tsc_transferhpo/Results/kfolds_RS_benchmarks/Running_' + dataset + '.json', lines = True)
            bool_series = df.duplicated(
                subset=['depth', 'use_residual', 'nb_filters', 'use_bottleneck', 'batch_size', 'kernel_size'])
            if len(bool_series[bool_series]) > 0:
                print(dataset, len(bool_series[bool_series]))

            # df2 = df[df.applymap(lambda x: x[0] if isinstance(x, list) else x).duplicated()]
            # print(dataset, len(df2))

        except Exception as e:
            print(dataset, e)
            pass


def run_missing_config(listed):
    run_folder = root_dir + '/Results/kfolds_RS_benchmarks'
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

    for dataset in listed:
        try:
            df = pd.read_json( root_dir + '/Results/kfolds_RS_benchmarks/Running_' + dataset + '.json', lines=True)
        except Exception as e:
            print(e)
            continue

        # if len(df) < 190:
        #     continue

        df['use_bottleneck'] = df['use_bottleneck'].map({'True': True, 'False': False, 'true': True, 'false': False})
        df['use_residual'] = df['use_residual'].map({'True': True, 'False': False, 'true': True, 'false': False})
        df = df.drop(
            columns=['dataset', 'acc', 'precision', 'recall', 'budget_ran', 'train_curve', 'val_curve', 'train_curve 1',
                     'val_curve 1', 'acc 1', 'precision 1', 'recall 1', 'train_curve 2',
                     'val_curve 2', 'acc 2', 'precision 2', 'recall 2', 'train_curve 3',
                     'val_curve 3', 'acc 3', 'precision 3', 'recall 3', 'train_curve 4',
                     'val_curve 4', 'acc 4', 'precision 4', 'recall 4', 'train_curve 5',
                     'val_curve 5', 'acc 5', 'precision 5', 'recall 5'])
        df_matrix = df.to_numpy()

        to_do = []
        i = 0
        for x in df_ref_matrix:
            for y in df_matrix:
                if (x == y).all():
                    i += 1
                    break
            else:
                to_do.append(x)

        print(to_do)

        if len(to_do) > 0:
            print('################################', dataset, '######################################################')

        for each in to_do:
            results_dir = root_dir + '/Results'
            run_name = 'kfolds_RS_benchmarks'
            run_folder = results_dir + '/' + run_name + '/'
            ind_m = np.all(df_ref_matrix == each, axis=1)
            iii = np.where(ind_m == True)[0][0]
            print(iii)
            print(df_ref.iloc[iii])
            config = {'depth': int(df_ref.iloc[iii]['depth'].astype(int)),
                      'nb_filters': int(df_ref.iloc[iii]['nb_filters'].astype(int)),
                      'kernel_size': int(df_ref.iloc[iii]['kernel_size'].astype(int)),
                      'batch_size': int(df_ref.iloc[iii]['batch_size'].astype(int)),
                      'use_residual': bool(df_ref.iloc[iii]['use_residual'].astype(bool)),
                      'use_bottleneck': bool(df_ref.iloc[iii]['use_bottleneck'].astype(bool))}
            print(config)
            for k, v in config.items():
                print(k, type(v))
            objective_renewed(config, dataset, run=run_name, n_splits=5, output_dir=run_folder)


def handling_json(given):
    for dataset in given:
        # target_file = r'E:\thesis_work\tsc_transferhpo\Results\kfolds_RS_benchmarks\RS_sampled_configs.json'
        query_file = r'E:\thesis_work\tsc_transferhpo\Results\kfolds_RS_benchmarks\Running_' + dataset + '.json'
        if os.path.exists(query_file):
            save_file = r'E:\thesis_work\tsc_transferhpo\Final_benchmarks\Running_' + dataset + '.json'
            if os.path.exists(save_file):
                os.remove(save_file)
            q_df = pd.read_json(query_file, lines=True)
            q_df = q_df.drop_duplicates(subset=['depth', 'nb_filters', 'batch_size', 'kernel_size', 'use_residual', 'use_bottleneck'])
            q_df = q_df.sort_values(by=['depth', 'nb_filters', 'batch_size', 'kernel_size', 'use_residual', 'use_bottleneck'])
            q_df = q_df.reset_index(drop=True)
            q_df.to_json(save_file)


# count()
# check_for_duplicates()
run_missing_config(['FordB'])
# create proper json file
# handling_json(datasets)
# os.system("shutdown /s /t 1")
