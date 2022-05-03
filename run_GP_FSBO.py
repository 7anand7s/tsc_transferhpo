import numpy as np
import os
import json
import argparse
import shutil
import random

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from all_functions import fsbo_running, extract_grid, results_dir, folders, fitting_smbo

parser = argparse.ArgumentParser()


def random_start_smbo(n, grid_matrix, acc_matrix, save_dir, iters_ran=0):
    X_initialize = []
    Y_initialize = []

    # for _ in range(n):
    for curr_inc in n:
        # d_size = grid_matrix.shape[0] - 1
        # curr_inc = np.random.randint(0, d_size)

        X_initialize.append(grid_matrix[curr_inc].astype(int))
        Y_initialize.append(float(acc_matrix[curr_inc]))

        with open(save_dir, 'a+') as f:
            json.dump({'config': grid_matrix[curr_inc].tolist(),
                       'accuracy': float(acc_matrix[curr_inc])
                       }, f)
            f.write("\n")

        grid_matrix = np.delete(grid_matrix, int(curr_inc), 0)
        acc_matrix = np.delete(acc_matrix, int(curr_inc), 0)

        iters_ran += 1

    return grid_matrix, acc_matrix, iters_ran, X_initialize, Y_initialize


if __name__ == '__main__':

    parser.add_argument('-n_init', type=int, default=5)
    parser.add_argument('-iters', type=int, default=50)
    parser.add_argument('-n_split', type=int, default=5)
    parser.add_argument('-fsbo_train', type=int, default=10000)
    parser.add_argument('-fsbo_tune', type=int, default=5000)

    args = parser.parse_args()
    n_warm_start = args.n_init
    n_iter = args.iters
    n_split = args.n_split
    fsbo_train = args.fsbo_train
    fsbo_tune = args.fsbo_tune

    rand_init = random.sample(range(0, 199), n_warm_start)

    # raise ValueError('podhum')

    # folder which contains the benchmarks
    run_folder = results_dir + '/kfolds_RS_benchmarks/'

    # ################################ BO-GP run #################################################

    # create directory to save results from the run
    gp_dir = results_dir + 'BO_GP_rand_init/'
    if not os.path.exists(gp_dir):
        os.mkdir(gp_dir)

    # Setting configuration for GP kernel
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52)

    for name in folders:

        read_dir = run_folder + '/Running_' + name + '.json'
        grid_m, acc_m = extract_grid(read_dir)
        dim = 6
        X_init = []
        Y_init = []

        if os.path.exists(gp_dir + 'GP_' + name + '.json'):
            shutil.rmtree(gp_dir)
            os.mkdir(gp_dir)

        saving_path = gp_dir + 'GP_' + name + '.json'
        grid_m, acc_m, iterations_ran, X_init, Y_init = random_start_smbo(rand_init, grid_m, acc_m, saving_path)

        # Initialize samples
        X_sample = np.array(X_init).reshape(-1, dim)
        Y_sample = np.array(Y_init).reshape(-1, 1)

        use_dir = gp_dir + 'GP_' + name + '.json'
        fitting_smbo(gpr, use_dir, iterations_ran, n_iter, grid_m, acc_m, X_sample, Y_sample)

    # ################################ BO-FSBO run #################################################

    # create directory to save results from the run
    fsbo_dir = results_dir + 'BO_FSBO_rand_init/'
    if not os.path.exists(fsbo_dir):
        os.mkdir(fsbo_dir)

    fsbo_running(rand_init, n_iter, folders, fsbo_dir, fsbo_train, fsbo_tune, transfer=False, cc=4)
