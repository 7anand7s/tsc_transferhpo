import numpy as np
import os
import json
import argparse
import pandas as pd
import shutil
import glob

import tqdm.std
from matplotlib import pyplot as plt
from all_functions import extract_grid, results_dir, folders

parser = argparse.ArgumentParser()


def hpo_rs():
    # create directory for RS
    work_dir = results_dir + 'RS_run/'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    for iter_data in folders:
        try:
            iterations_ran = 0

            read_dir = results_dir + 'kfolds_RS_benchmarks/Running_' + iter_data + '.json'
            grid_m, acc_m = extract_grid(read_dir)

            if os.path.exists(work_dir + 'RS_' + iter_data + '.json'):
                shutil.rmtree(work_dir)
                os.mkdir(work_dir)

            while iterations_ran < n_iter:
                data_size = grid_m.shape[0] - 1
                curr_inc = np.random.randint(0, data_size)

                with open(work_dir + 'RS_' + iter_data + '.json', 'a+') as f:
                    json.dump({'config': grid_m[curr_inc].tolist(),
                               'accuracy': float(acc_m[curr_inc])
                               }, f)
                    f.write("\n")

                grid_m = np.delete(grid_m, int(curr_inc), 0)
                acc_m = np.delete(acc_m, int(curr_inc), 0)

                iterations_ran += 1
        except Exception as e:
            print(e)
            raise ValueError('fix the exception : ', e)


if __name__ == '__main__':

    parser.add_argument('-rs', type=int, default=0)
    parser.add_argument('-iters', type=int, default=50)

    np.seterr(divide='ignore', invalid='ignore')

    args = parser.parse_args()
    rs_run = bool(args.rs)
    n_iter = args.iters

    if rs_run:
        hpo_rs()

    # Plots
    names = [#'RS_run',
             # 'BO_GP_rand_init',
             # 'BO_FSBO_rand_init',
             # 'BO_GP_topn_init_fawaz',
             # 'BO_GP_topn_init_anand',
             # 'BO_GP_tidal_init_fawaz',
             # 'BO_GP_tidal_init_anand',
             # 'BO_GP_diverse_init_fawaz',
             'BO_GP_diverse_init_anand',
             # 'BO_GP_diverse_init_smfo',
             # 'BO_FSBO_topn_init_fawaz',
             # 'BO_FSBO_topn_init_anand',
             # 'BO_FSBO_tidal_init_fawaz',
             # 'BO_FSBO_tidal_init_anand',
             # 'BO_FSBO_diverse_init_fawaz',
             'BO_FSBO_diverse_init_anand',
             ]

    Big_Y = []
    for hehe in range(len(names)):
        Big_Y.append([])

    ranks_y = []

    pbar = tqdm.tqdm(total=len(folders), desc='Plotting graphs for each dataset')

    for name in folders:

        regrets_y = []

        plt.clf()
        dfs = []

        for each_folders in names:
            a = results_dir + each_folders + '/'
            b = '*' + name + '.json'
            for file in glob.glob(os.path.join(a,b)):
                # print(file)
                dfs.append(pd.read_json(file, lines=True))

        dfs_orig = pd.read_json(results_dir + 'kfolds_RS_benchmarks/Running_' + name + '.json', lines=True)

        dfs_acc = []
        for df in dfs:
            df = df.head(50)
            dfs_acc.append(df['accuracy'].values if 'accuracy' in df else df['acc'].values)

        data = np.array(dfs_acc)

        y = []
        for each_x in data:
            y1 = np.zeros(each_x.shape)
            y1[0] = each_x[0]
            for li in range(1, len(each_x)):
                y1[li] = max(each_x[:li])

            y.append(y1)

        orig_acc = dfs_orig['accuracy'].values if 'accuracy' in dfs_orig else dfs_orig['acc'].values
        maximum_val = max(orig_acc)
        y_star = np.ones(len(y[0])) * maximum_val
        for iter_y in y:
            regrets_y.append(y_star - iter_y)

        calculate_df = pd.DataFrame(regrets_y)

        rankings = calculate_df.rank(axis=0)

        for plot_idx in range(len(regrets_y)):
            plt.plot(np.arange(len(rankings.iloc[plot_idx])), rankings.iloc[plot_idx], label=names[plot_idx])
            Big_Y[plot_idx].append(rankings.iloc[plot_idx])

        plt.legend()
        plt.savefig(results_dir + 'trial_Graphs/' + name + '.png')

        pbar.update()

    plt.clf()
    for iter_index in range(len(Big_Y)):
        temp_v = sum(Big_Y[iter_index]) / len(Big_Y[iter_index])
        plt.plot(np.arange(len(temp_v)), temp_v, label=names[iter_index])

    # plt.legend(loc='upper left')
    plt.legend()
    plt.savefig(results_dir + 'trial_Graphs/00_AVERAGED_k2.png')
