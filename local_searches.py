import sys
import pandas as pd
import numpy as np
from utils.utils import root_dir
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from smbo import propose_location, expected_improvement
from matplotlib import pyplot as plt
# from utils.constants import UNIVARIATE_DATASET_NAMES
# from smac.configspace import ConfigurationSpace
# from smac.facade.smac_bb_facade import SMAC4BB
# from smac.scenario.scenario import Scenario
# from smac.optimizer.acquisition import EI
# from functools import partial
# from utils.utils import read_all_datasets, transform_labels
# from utils.constants import UNIVARIATE_DATASET_NAMES as dataset_names
# from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
# from utils.utils import save_logs
# from utils.utils import calculate_metrics
# from utils.utils import save_test_duration
# import time
# import json
# import tensorflow
# import tensorflow as tf
# import os
# import sklearn
import numpy as np


# import ConfigSpace.hyperparameters as CSH
# import logging


def objective(ind):
    SMBO.append(df2['acc'].loc[ind])
    return df2['acc'].loc[ind]


if sys.argv[1] == 'compare_searching':
    temp = 1
    n_iter = 50

    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52)

    # Overall measurements needed
    acc_mat_y1 = []
    acc_mat_y2 = []
    reg_mat_y1 = []
    reg_mat_y2 = []
    rank_mat_y1 = []
    rank_mat_y2 = []
    arank_mat_y1 = []
    arank_mat_y2 = []

    folders = ['BeetleFly', 'CBF', 'CinC_ECG_torso']
    # ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
    #  'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
    #  'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
    #  'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
    #  'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'FaceAll']
    for c, name in enumerate(folders):
        RS = []
        SMBO = []
        iterations_ran = 0

        print(name)
        # E:\thesis_work\tsc_transferhpo\Results\benchmark

        filee_path1 = root_dir + '/Results/benchmark/' + name + '/Running_' + name + '.json'

        columns = ["dataset", "depth", "nb_filters", "batch_size", "kernel_size", "use_residual", "use_bottleneck",
                   "acc", "precision", "recall", "budget_ran", "duration"]

        # try:
        df = pd.read_csv(filee_path1, names=columns)
        # except:
        #     print("****Data not found****", name)
        #     continue

        # print(df)
        df['dataset'] = df['dataset'].str.slice(start=12)
        df['depth'] = df['depth'].str.slice(start=9).astype(int)
        df['acc'] = df['acc'].str.slice(start=7).astype(float)
        df['batch_size'] = df['batch_size'].str.slice(start=14).astype(int)
        df['kernel_size'] = df['kernel_size'].str.slice(start=15).astype(int)
        df['use_residual'] = df['use_residual'].str.slice(start=16)
        df['use_bottleneck'] = df['use_bottleneck'].str.slice(start=19)
        df['nb_filters'] = df['nb_filters'].str.slice(start=14).astype(int)
        df['precision'] = df['precision'].str.slice(start=13).astype(float)
        df['recall'] = df['recall'].str.slice(start=10).astype(float)
        df['budget_ran'] = df['budget_ran'].str.slice(start=15).astype(int)
        df['duration'] = df['duration'].str.slice(start=12, stop=-1).astype(float)

        if df.shape[0] < 200:
            print("----not done----", name)
            continue
        temp += 1
        print(temp)
        df2 = df.copy()

        # ---------------------------------------------------------------------------------
        # RS in 1D
        indices = np.arange(0, 200)
        for _ in range(n_iter):
            x0 = np.random.choice(indices)
            RS.append(df2['acc'].loc[x0])
            indices = np.delete(indices, np.where(indices == x0))
            df2 = df2.drop(x0)

        # ---------------------------------------------------------------------------------
        # SMBO in smac
        # cs = ConfigurationSpace()
        #
        # index = CSH.UniformIntegerHyperparameter(name='index', lower=0,
        #                                          upper=199, log=False)
        #
        # cs.add_hyperparameters([index])
        #
        # scenario = Scenario({
        #     'abort_on_first_run_crash': False,
        #     'run_obj': 'quality',  # we optimize quality (alternative to runtime)
        #     # 'wallclock-limit': 3600,  # max duration to run the optimization (in seconds)
        #     'runcount-limit': n_iter,  # Max number of function evaluations (the more the better)
        #     'cs': cs,  # configuration space
        #     'deterministic': 'true',
        #     'limit_resources': True,  # Uses pynisher to limit memory and runtime
        #     # Alternatively, you can also disable this.
        #     # Then you should handle runtime and memory yourself in the TA
        #     # 'cutoff': 30,  # runtime limit for target algorithm
        #     # 'memory_limit': 3072,  # adapt this to reasonable value for your hardware
        # })
        #
        # # To optimize, we pass the function to the SMAC-object
        # smac = SMAC4BB(
        #     scenario=scenario,
        #     model_type='gp',
        #     acquisition_function=EI,
        #     rng=np.random.RandomState(42),
        #     tae_runner=objective
        # )
        #
        # # Start optimization
        # try:
        #     incumbent = smac.optimize()
        # finally:
        #     incumbent = smac.solver.incumbent
        #
        # opt_config = incumbent.get_dictionary()
        #
        # print(opt_config)

        # ---------------------------------------------------------------------------------
        # SMBO in 1D
        dim = 1
        X_init = []
        Y_init = []
        indices = np.arange(0, 200)
        for _ in range(1):
            x0 = np.random.choice(indices)
            X_init.append(x0)
            Y_init.append(df['acc'].loc[x0])
            SMBO.append(df['acc'].loc[x0])
            df = df.drop(x0)
            indices = np.delete(indices, np.where(indices == x0))
            iterations_ran += 1

        # Initialize samples
        X_sample = np.array(X_init).reshape(-1, dim)
        Y_sample = np.array(Y_init).reshape(-1, 1)

        while iterations_ran < n_iter:
            # Update Gaussian process with existing samples
            gpr.fit(X_sample, Y_sample)

            # Obtain next sampling point from the acquisition function (expected_improvement)
            X_next, loc = propose_location(expected_improvement, X_sample, Y_sample, gpr, indices)

            # if X_next is None:
            #     continue

            Y_next = np.array(df['acc'].loc[int(X_next)])
            SMBO.append(df['acc'].loc[int(X_next)])

            X_val = int(X_next)
            Y_val = Y_next.reshape(-1, 1)

            # Add sample to previous samples
            X_sample = np.vstack((X_sample, X_val))
            Y_sample = np.vstack((Y_sample, Y_val))

            df = df.drop(int(X_next))
            indices = np.delete(indices, np.where(indices == int(X_next)))

            iterations_ran += 1

            # Plot samples, surrogate function, noise-free objective and next sampling location
            # plt.subplot(n_iter, 2, 2 * i + 1)
            # plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i == 0)
            # plt.title(f'Iteration {i + 1}')
            #
            # plt.subplot(n_iter, 2, 2 * i + 2)
            # plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i == 0)

        # ---------------------------------------------------------------------------------
        # accuracy graph
        # curve 1
        matrx00 = np.zeros(len(SMBO))
        for i1, x1 in enumerate(SMBO):
            matrx00[i1] = x1
        x1 = (matrx00 - np.min(matrx00)) / (np.max(matrx00) - np.min(matrx00))
        x1[np.isnan(x1)] = 0
        y1 = np.zeros(x1.shape)
        for li1 in range(1, len(x1)):
            y1[li1] = max(x1[:li1])

        # curve 2
        matrx01 = np.zeros(len(RS))
        for i2, x2 in enumerate(RS):
            matrx01[i2] = x2
        x2 = (matrx01 - np.min(matrx01)) / (np.max(matrx01) - np.min(matrx01))
        x2[np.isnan(x2)] = 1
        y2 = np.zeros(x2.shape)
        for li2 in range(1, len(x2)):
            y2[li2] = max(x2[:li2])

        acc_mat_y1.append(y1)
        acc_mat_y2.append(y2)

        # ---------------------------------------------------------------------------------
        # Regret graph
        maximum_val = max(df['acc'])

        y_star = np.ones(len(y2)) * maximum_val

        reg_y1 = y_star - y1
        reg_y2 = y_star - y2

        reg_mat_y1.append(reg_y1)
        reg_mat_y2.append(reg_y2)

        # ---------------------------------------------------------------------------------
        # Ranking graph
        y1_theta = []
        y2_theta = []
        for a, b in zip(y1, y2):
            if a > b:
                y1_theta.append(1)
                y2_theta.append(2)
            elif a == b:
                y1_theta.append(1.5)
                y2_theta.append(1.5)
            elif a < b:
                y1_theta.append(2)
                y2_theta.append(1)

        rank_mat_y1.append(y1_theta)
        rank_mat_y2.append(y2_theta)

        # ---------------------------------------------------------------------------------
        # Average Ranking graph
        y1_omega = [y1_theta[0]]
        y2_omega = [y2_theta[0]]
        for count, (a, b) in enumerate(zip(y1, y2)):
            y1_omega.append(sum(y1_theta[:count + 1]) / (count + 1))
            y2_omega.append(sum(y2_theta[:count + 1]) / (count + 1))

        arank_mat_y1.append(y1_omega)
        arank_mat_y2.append(y2_omega)

    # ---------------------------------------------------------------------------------
    # Averaging the mass results into a single matrix
    acc_mat_y1 = np.average(np.vstack(acc_mat_y1), axis=0)
    acc_mat_y2 = np.average(np.vstack(acc_mat_y2), axis=0)
    reg_mat_y1 = np.average(np.vstack(reg_mat_y1), axis=0)
    reg_mat_y2 = np.average(np.vstack(reg_mat_y2), axis=0)
    rank_mat_y1 = np.average(np.vstack(rank_mat_y1), axis=0)
    rank_mat_y2 = np.average(np.vstack(rank_mat_y2), axis=0)
    arank_mat_y1 = np.average(np.vstack(arank_mat_y1), axis=0)
    arank_mat_y2 = np.average(np.vstack(arank_mat_y2), axis=0)

    # ---------------------------------------------------------------------------------
    # Starting plots
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(np.arange(len(acc_mat_y1)), acc_mat_y1, 'b', label='SMBO')
    axs[0, 0].plot(np.arange(len(acc_mat_y2)), acc_mat_y2, 'g', label='RS')
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].legend()
    axs[0, 1].plot(np.arange(len(reg_mat_y1)), reg_mat_y1, 'b', label='SMBO')
    axs[0, 1].plot(np.arange(len(reg_mat_y2)), reg_mat_y2, 'g', label='RS')
    axs[0, 1].set_title('Regret')
    axs[0, 1].legend()
    axs[1, 0].plot(np.arange(len(rank_mat_y1)), rank_mat_y1, 'b', label='SMBO')
    axs[1, 0].plot(np.arange(len(rank_mat_y2)), rank_mat_y2, 'g', label='RS')
    axs[1, 0].set_title('Ranking')
    axs[1, 0].legend()
    axs[1, 1].plot(np.arange(len(arank_mat_y1)), arank_mat_y1, 'b', label='SMBO')
    axs[1, 1].plot(np.arange(len(arank_mat_y2)), arank_mat_y2, 'g', label='RS')
    axs[1, 1].set_title(' Average Ranking')
    axs[1, 1].legend()

    # ---------------------------------------------------------------------------------
    plt.savefig("mygraph.png")
