import pandas as pd
import numpy as np
import os
import argparse

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from all_functions import results_dir, fsbo_running, folders, extracting_indices, dist_smfo, dist_anand, dist_fawaz, \
    extract_grid, dissolving_grid, fitting_smbo

parser = argparse.ArgumentParser()


if __name__ == '__main__':

    parser.add_argument('BO_type', choices=['GP', 'FSBO'], default='GP')
    parser.add_argument('ws_type', choices=['topn', 'tidal', 'diverse'], default='diverse')
    parser.add_argument('dist_type', choices=['smfo', 'fawaz', 'anand'], default='anand')
    parser.add_argument('-n_init', type=int, default=5)
    parser.add_argument('-iters', type=int, default=50)
    parser.add_argument('-cc', type=int, default=None)
    parser.add_argument('-n_split', type=int, default=5)
    parser.add_argument('-fsbo_train', type=int, default=10000)
    parser.add_argument('-fsbo_tune', type=int, default=5000)

    args = parser.parse_args()
    n_warm_start = args.n_init
    n_iter = args.iters
    cc = args.cc
    BO_type = args.BO_type
    ws_type = args.ws_type
    dist_type = args.dist_type
    n_split = args.n_split
    fsbo_train = args.fsbo_train
    fsbo_tune = args.fsbo_tune

    # choose the distance method ****************************************
    if dist_type == 'fawaz':
        dist_df = dist_fawaz
    elif dist_type == 'anand':
        dist_df = dist_anand
    elif dist_type == 'smfo':
        dist_df = dist_smfo
    else:
        raise ValueError('Specify the distance type')

    # choose how initial configs are chosen -----------------------------
    if ws_type == 'topn':
        method = 'm1'
    elif ws_type == 'tidal':
        method = 'm2'
    elif ws_type == 'diverse':
        method = 'm3'
    else:
        raise ValueError('Specify how initial configs are chosen')

    # create sorted benchmarks before running any TF algorithm
    if not os.path.exists(results_dir + 'trial_benchmark_sorted/'):
        dissolving_grid(folders)

    # ################################ BO-GP run #################################################
    if BO_type == 'GP':

        # Setting configuration for GP kernel
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gpr = GaussianProcessRegressor(kernel=m52)

        # create directory to save results from the run
        gp_dir = results_dir + 'BO_GP_' + ws_type + '_init_' + dist_type + '/'
        if not os.path.exists(gp_dir):
            os.mkdir(gp_dir)

        # folder which contains the benchmarks
        run_folder = results_dir + '/kfolds_RS_benchmarks/'

        for loop1, ind_c in zip(folders, range(len(dist_df))):
            # iterations_ran = 0
            source_d = dist_df['K_1'][ind_c]

            dim = 6
            read_dir = results_dir + 'kfolds_RS_benchmarks' + '/Running_' + loop1 + '.json'
            grid_m2, acc_m2 = extract_grid(read_dir)

            if not os.path.exists(results_dir + 'trial_benchmark_sorted' + '/RS_' + source_d + '.json'):
                raise ValueError('Create sorted benchmarks and run again')

            df3 = pd.read_json(results_dir + 'trial_benchmark_sorted' + '/RS_' + source_d + '.json', lines=True)

            savin_dir = gp_dir + loop1 + '.json'
            X_init, Y_init, grid_m2, acc_m2, iterations_ran \
                = extracting_indices(n_warm_start, df3, grid_m2, acc_m2, savin_dir, method=method)

            # Initialize samples
            X_sample = np.array(X_init).reshape(-1, dim)
            Y_sample = np.array(Y_init).reshape(-1, 1)

            fitting_smbo(gpr, savin_dir, iterations_ran, n_iter, grid_m2, acc_m2, X_sample, Y_sample)

    # ################################ BO-FSBO run #################################################

    if BO_type == 'FSBO':

        # create directory to save results from the run
        fsbo_dir = results_dir + 'BO_FSBO_' + ws_type + '_init_' + dist_type + '/'
        if not os.path.exists(fsbo_dir):
            os.mkdir(fsbo_dir)

        # run FSBO with the required configs
        fsbo_running(n_warm_start, n_iter, folders, fsbo_dir, fsbo_train, fsbo_tune,
                     transfer=True, tf_method=method, dist_data=dist_df, cc=cc)
