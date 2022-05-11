# [ 'ChlorineConcentration', 'CinC_ECG_torso', 'ElectricDevices',
# 'FordA', 'FordB', 'HandOutlines', 'InsectWingbeatSound',
# 'MALLAT', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2',
# 'PhalangesOutlinesCorrect', 'Phoneme', 'StarLightCurves',
# 'Two_Patterns', 'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
# 'uWaveGestureLibrary_Z', 'wafer', 'yoga']

######################################################################################
# Currently running in cluster

# UWaveGestureLibraryAll 140       - ✓ cluster 4
# MALLAT 187                        - ✓ cluster 3
# ElectricDevices                   - cluster cpu
# uWaveGestureLibrary_Y 183 - ✓ pc
# uWaveGestureLibrary_Z 182 - ✓ cluster cpu


# # CinC_ECG_torso HandOutlines MALLAT FordA Two_Patterns yoga
#
# import pandas as pd
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# # pd_df = pd.read_json(r'E:\thesis_work\tsc_transferhpo\Results\kfolds_RS_benchmarks\Running_ArrowHead.json', lines=True)
# # print(pd_df)
#
# # FSBO diverse fawaz - 111003 - jobID 14716396
# # FSBO diverse anand_agg4 - 111000 - jobID 14716398
# # FSBO topn anand_agg4 - 111001 - jobID 14716401
# # FSBO topn fawaz - 111002 - jobID 14716428
#
import pandas as pd
import json
import os
import numpy as np

from k_folds import objective_renewed
from utils.utils import root_dir
from utils.constants import UNIVARIATE_DATASET_NAMES as datasets


# from all_functions import folders

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
            df = pd.read_json(root_dir + '/Results/kfolds_RS_benchmarks/Running_' + dataset + '.json', lines=True)
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
            # print(iii)
            # print(df_ref.iloc[iii])
            config = {'depth': int(df_ref.iloc[iii]['depth'].astype(int)),
                      'nb_filters': int(df_ref.iloc[iii]['nb_filters'].astype(int)),
                      'kernel_size': int(df_ref.iloc[iii]['kernel_size'].astype(int)),
                      'batch_size': int(df_ref.iloc[iii]['batch_size'].astype(int)),
                      'use_residual': bool(df_ref.iloc[iii]['use_residual'].astype(bool)),
                      'use_bottleneck': bool(df_ref.iloc[iii]['use_bottleneck'].astype(bool))}
            # print('--6--')
            if int(config['depth']) == 6:
                print(iii)
            # print('--3--')
            # if int(df_ref.iloc[iii]['depth'].astype(int)) == 3:
            #     print(iii)

            # print(config)
            # for k, v in config.items():
            #     print(k, type(v))
            # objective_renewed(config, dataset, run=run_name, n_splits=5, output_dir=run_folder)


def check_for_duplicates():
    for dataset in datasets:
        try:
            df = pd.read_json(
                'E:/thesis_work/tsc_transferhpo/Results/kfolds_RS_benchmarks/Running_' + dataset + '.json', lines=True)

            print(dataset, len(df))
            bool_series = df.duplicated(
                subset=['depth', 'use_residual', 'nb_filters', 'use_bottleneck', 'batch_size', 'kernel_size'])
            if len(bool_series[bool_series]) > 0:
                print("duplicates in ", dataset, len(bool_series[bool_series]))

            # df2 = df[df.applymap(lambda x: x[0] if isinstance(x, list) else x).duplicated()]
            # print(dataset, len(df2))

        except Exception as e:
            print(dataset, e)
            pass


# run_missing_config(['ElectricDevices'])
# os.system("shutdown /s /t 1")
check_for_duplicates()
# ElectricDevices with depth == 9
# 0 2 3 9 12 13 17 20 27 32 33 40 41 42 43 49 51 53 57 58 62 65 66 68 73 77 78 80 81 82 84 91 96 105 107 109 111 113 115 116 118 123 125 126 129 134 136 139 140 148 149 154 156 158 161 162 172 174 178 186 190 199

# #!/bin/bash
# #MSUB -l walltime=24:00:00
# #MSUB -l nodes=1:ppn=2:gpus=1
# #MSUB -l pmem=4gb
# #MSUB -v ID
# #MSUB -N trhpo_run${ID}
# #MSUB -v VARIABLE
# #MSUB -v ITER
#
# source /home/fr/fr_fr/fr_aa367/anaconda3/bin/activate trhpo
#
# python /home/fr/fr_fr/fr_aa367/tsc_transferhpo/create_BigData_benchmark.py -nargs ${VARIABLE} -iter$
#
#
#
# #!/bin/bash
# #MSUB -l walltime=20:00:00
# #MSUB -l nodes=1:ppn=2
# #MSUB -l pmem=4gb
# #MSUB -v ID
# #MSUB -N fsbo_run
# #MSUB -v BO_type
# #MSUB -v ws_type
# #MSUB -v dist_type
# #MSUB -v cc
# #MSUB -v fsbo_train
# #MSUB -v fsbo_tune
# #MSUB -v frozen
#
#
# source /home/fr/fr_fr/fr_aa367/anaconda3/bin/activate trhpo
#
# python /home/fr/fr_fr/fr_aa367/tsc_transferhpo/transfer_learning.py FSBO
# ${ws_type}  ${dist_type} -cc ${cc} -fsbo_train ${fsbo_train} -fsbo_tune ${fsbo_tune}
# -freeze ${frozen}
#
#
# # -v ID=${ID} -v ws_type=${ws_type} -v dist_type=${dist_type} -v cc=${cc} -v fsbo_train=${fsbo_train} -v fsbo_tune=${fsbo_tune} -v =${frozen}
#
# ID=9991200
# DATASET=ElectricDevices
# # for ((i=199; i>=150; i--))
# for i in 0 2 3 9 12 13 17 20 27 32 33 40 41 42 43 49 51 53 57 58 62 65 66 68 73 77 78 80 81 82 84 $
# # for i in 29
# do
#       jobID=$(msub mymoabfile_cpu.moab -v ID=${ID} -v VARIABLE=${DATASET} -v ITER=${i})
#       echo "{\"jobID\":${jobID}, \"ID\":${ID} ${DATASET}, \"iter_count\":${i}}"
#       let ID++
# done

# #!/bin/bash
# #MSUB -l walltime=96:00:00
# #MSUB -l nodes=1:ppn=3
# #MSUB -l pmem=6gb
# #MSUB -v ID
# #MSUB -N trhpo_run${ID}
# #MSUB -v VARIABLE
# #MSUB -v ITER
#
# source /home/fr/fr_fr/fr_aa367/anaconda3/bin/activate trhpo
#
# python /home/fr/fr_fr/fr_aa367/tsc_transferhpo/main.py create_BigData_benchmark.py -nargs ${VARIABLE} -iters ${ITER}

# {"jobID":
# 14783368, "ID":99924000 ElectricDevices, "iter_count":1}
# {"jobID":
# 14783369, "ID":99924001 ElectricDevices, "iter_count":5}
# {"jobID":
# 14783370, "ID":99924002 ElectricDevices, "iter_count":8}
# {"jobID":
# 14783371, "ID":99924003 ElectricDevices, "iter_count":16}
# {"jobID":
# 14783372, "ID":99924004 ElectricDevices, "iter_count":21}
# {"jobID":
# 14783373, "ID":99924005 ElectricDevices, "iter_count":28}
# {"jobID":
# 14783374, "ID":99924006 ElectricDevices, "iter_count":31}
# {"jobID":
# 14783375, "ID":99924007 ElectricDevices, "iter_count":34}
# {"jobID":
# 14783376, "ID":99924008 ElectricDevices, "iter_count":45}
# {"jobID":
# 14783377, "ID":99924009 ElectricDevices, "iter_count":46}
# {"jobID":
# 14783378, "ID":99924010 ElectricDevices, "iter_count":52}
# {"jobID":
# 14783379, "ID":99924011 ElectricDevices, "iter_count":90}
# {"jobID":
# 14783380, "ID":99924012 ElectricDevices, "iter_count":98}
# {"jobID":
# 14783381, "ID":99924013 ElectricDevices, "iter_count":104}
# {"jobID":
# 14783382, "ID":99924014 ElectricDevices, "iter_count":121}
# {"jobID":
# 14783383, "ID":99924015 ElectricDevices, "iter_count":133}
# {"jobID":
# 14783384, "ID":99924016 ElectricDevices, "iter_count":146}
# {"jobID":
# 14783385, "ID":99924017 ElectricDevices, "iter_count":153}
# {"jobID":
# 14783386, "ID":99924018 ElectricDevices, "iter_count":157}
# {"jobID":
# 14783387, "ID":99924019 ElectricDevices, "iter_count":164}
# {"jobID":
# 14783388, "ID":99924020 ElectricDevices, "iter_count":167}
# {"jobID":
# 14783389, "ID":99924021 ElectricDevices, "iter_count":169}
# {"jobID":
# 14783390, "ID":99924022 ElectricDevices, "iter_count":170}
# {"jobID":
# 14783391, "ID":99924023 ElectricDevices, "iter_count":171}
# {"jobID":
# 14783392, "ID":99924024 ElectricDevices, "iter_count":173}
# {"jobID":
# 14783393, "ID":99924025 ElectricDevices, "iter_count":176}
# {"jobID":
# 14783394, "ID":99924026 ElectricDevices, "iter_count":181}
# {"jobID":
# 14783395, "ID":99924027 ElectricDevices, "iter_count":183}
# {"jobID":
# 14783396, "ID":99924028 ElectricDevices, "iter_count":187}
# {"jobID":
# 14783397, "ID":99924029 ElectricDevices, "iter_count":189}
# {"jobID":
# 14783398, "ID":99924030 ElectricDevices, "iter_count":192}
# {"jobID":
# 14783399, "ID":99924031 ElectricDevices, "iter_count":196}
