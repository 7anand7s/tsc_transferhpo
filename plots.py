import tensorflow as tf
import tensorflow
import pandas as pd
import json
import numpy as np
import os
# from main_f import objective
import matplotlib.pyplot as plt
from utils.constants import UNIVARIATE_DATASET_NAMES, UNIVARIATE_ARCHIVE_NAMES

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
matrx21 = np.zeros([200, 85])
# matrx22 = np.zeros([40, 18])
# matrx23 = np.zeros([40, 18])
# matrx31 = np.zeros([40, 18])
# matrx32 = np.zeros([40, 18])
# matrx33 = np.zeros([40, 18])
count = 0

avv_t = []
avv_v = []
avv_t1 = []
avv_v1 = []
for names in [ '50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
                            'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
                            'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
                            'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                            'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
                            'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines']:

    print(names)
    filee_path1 = 'E:/thesis_work/tsc_transferhpo/Results/kfolds_with_ES_500_50_001/Running_' + names + '.json'
    filee_path2 = 'E:/thesis_work/tsc_transferhpo/Results/kfolds/Running_' + names + '.json'
    # filee_path3 = '/home/anand7s/Documents/transfer_hpo/bigdata18-master/configs_folder/smbo_gp_' + names + '.json'

    columns = ["dataset", "depth", "nb_filters", "batch_size", "kernel_size", "use_residual", "use_bottleneck",
               "acc", "precision", "recall", "budget_ran", "duration", "train_curve", "val_curve"]

    try:
        df = pd.read_json(filee_path1, lines=True)
        df2 = pd.read_json(filee_path2, lines=True)
    except:
        print("****Data not found****", names)
        continue
    # print(df)
    # df['dataset'] = df['dataset'].astype(str).str.slice(start=12)
    # df['depth'] = df['depth'].astype(str).str.slice(start=9).astype(int)
    # df['acc'] = df['acc'].astype(str).str.slice(start=7).astype(float)
    # df['batch_size'] = df['batch_size'].astype(str).str.slice(start=14).astype(int)
    # df['kernel_size'] = df['kernel_size'].astype(str).str.slice(start=15).astype(int)
    # df['use_residual'] = df['use_residual'].astype(str).str.slice(start=16)
    # df['use_bottleneck'] = df['use_bottleneck'].astype(str).str.slice(start=19)
    # df['nb_filters'] = df['nb_filters'].astype(str).str.slice(start=14).astype(int)
    # df['precision'] = df['precision'].astype(str).str.slice(start=13).astype(float)
    # df['recall'] = df['recall'].astype(str).str.slice(start=10).astype(float)
    # df['budget_ran'] = df['budget_ran'].astype(str).str.slice(start=15).astype(int)
    # df['train_curve'] = df['train_curve'].astype(str).str.slice(start=16).astype(list)
    # df['val_curve'] = df['val_curve'].astype(str).str.slice(start=14).astype(list)
    #
    # print(df)
    x = np.mean(df['acc'])
    y = np.mean(df2['acc'])
    for train, val, train2, val2 in zip(df['train_curve'], df['val_curve'], df2['train_curve'], df2['val_curve']):
        avv_t.append(train)
        avv_v.append(val)
        avv_t1.append(train2)
        avv_v1.append(val2)
        plt.plot(np.arange(len(train)), train, 'r', label='train_curve')
        plt.plot(np.arange(len(val)), val, 'b', label='validation_curve, Accuracy: %f' %x)
        plt.plot(np.arange(len(train2)), train2, 'r:', label='train_curve 2')
        plt.plot(np.arange(len(val2)), val2, 'b:', label='validation_curve 2, Accuracy: %f' %y)
        plt.title(names)
        plt.legend()
        plt.show()
t = np.average(avv_t, axis=0)
v = np.average(avv_v, axis=0)
plt.plot(np.arange(len(t)), t, 'r', label='train_curve')
plt.plot(np.arange(len(v)), v, 'b', label='validation_curve')
# plt.axvline(x=v[np.where(v==min(v))])
plt.title("average")
plt.legend()
plt.show()

#     # print(df['acc'])
#     if df.shape[0] < 200:
#         print("----not done----", names)
#         continue
#
#     print(df.shape[0])
#     df = df.tail(200)
#     print("renewed", df.shape[0])
#
#
#     # df = pd.read_json(filee_path1)
#     # df2 = pd.read_json(filee_path2)
#     # df3 = pd.read_json(filee_path3)
#
#     # curve 1
#     matrx00 = np.zeros(len(df['acc']))
#     for i1, x1 in enumerate(df['acc']):
#         matrx00[i1] = x1
#     x1 = (matrx00 - np.min(matrx00)) / (np.max(matrx00) - np.min(matrx00))
#     x1[np.isnan(x1)] = 0
#     y1 = np.zeros(x1.shape)
#     for li in range(1, len(x1)):
#         y1[li] = max(x1[:li])
#     m1 = np.arange(len(x1))
#
#     # curve 2
#     # matrx01 = np.zeros(len(df2['acc']))
#     # for i2, x2 in enumerate(df2['acc']):
#     #     matrx01[i2] = x2
#     # x2 = (matrx01 - np.min(matrx01)) / (np.max(matrx01) - np.min(matrx01))
#     # y2 = np.zeros(x2.shape)
#     # for li in range(1, len(x2)):
#     #     y2[li] = max(x2[:li])
#     # m2 = np.arange(len(x2))
#     #
#     # # curve 2
#     # matrx03 = np.zeros(len(df3['acc']))
#     # for i3, x3 in enumerate(df3['acc']):
#     #     matrx03[i3] = x3
#     # x3 = (matrx03 - np.min(matrx03)) / (np.max(matrx03) - np.min(matrx03))
#     # y3 = np.zeros(x3.shape)
#     # for li in range(1, len(x3)):
#     #     y3[li] = max(x3[:li])
#     # m3 = np.arange(len(x3))
#
#     # x1 = x1[:23]
#     # x2 = x2[:23]
#     # x3 = x3[:23]
#     # if len(y1) < len(y2):
#     #     y2 = y2[:len(y1)]
#     # if len(y1) > len(y2):
#     #     y1 = y1[:len(y2)]
#     # if len(y2) > len(y3):
#     #     y2 = y2[:len(y3)]
#     # if len(y3) > len(y2):
#     #     y3 = y3[:len(y2)]
#     # if len(y1) < len(y2):
#     #     y2 = y2[:len(y1)]
#     # if len(y1) > len(y2):
#     #     y1 = y1[:len(y2)]
#
#     # print( len(y2), len(y3))
#
#     # maximum_val = max(max(y2), max(y3))
#
#     # y_star = np.ones(len(y2)) * maximum_val
#
#     # rank_y1 = y_star - y1
#     # rank_y2 = y_star - y2
#     # rank_y3 = y_star - y3
#     # rank2_y1 = rank_y1.copy()
#     # rank2_y2 = rank_y2.copy()
#     # rank2_y3 = rank_y3.copy()
#     # # print(x1, y_star, y1, y2)
#     # # for haha, hehe in zip(rank_y1, rank2_y1):
#     # #     print(haha, hehe)
#     # avg1 = 0
#     # avg2 = 0
#     # avg3 = 0
#     # # print(rank_y1[0], rank2_y1[0])
#     # for ii, yy in enumerate(rank_y2):
#     #
#     #     if ii==0:
#     #         continue
#     #     avg1 = (avg1 + rank_y1[ii]) / 2
#     #     rank2_y1[ii] = avg1
#     #     avg2 = (avg2 + rank_y2[ii]) / 2
#     #     rank2_y2[ii] = avg2
#     #     avg3 = (avg3 + rank_y3[ii]) / 2
#     #     rank2_y3[ii] = avg3
#     #     # print(rank_y1[ii], rank2_y1[ii])
#     #     # rank2_y2[ii] = np.average(rank2_y2[:ii+1])
#     #     # rank2_y3[ii] = np.average(rank2_y3[:ii + 1])
#     #     # print(rank_y1[ii], rank2_y1[ii])
#
#     # print(y1)
#     # for haha, hehe in zip(rank_y1, rank2_y1):
#     #     print(haha, hehe)
#     # plt.clf()
#     # plt.plot(np.arange(len(rank_y1)), y1, 'r', label='tf')
#     # plt.plot(np.arange(len(rank_y2)), y2, 'g', label='RS')
#     # plt.plot(np.arange(len(m1)), y1, 'b', label='SMBO')
#     # plt.plot(np.arange(len(rank2_y1)), rank2_y1, 'r--', label='tf')
#     # plt.plot(np.arange(len(rank2_y2)), rank2_y2, 'g-.', label='RS1')
#     # plt.plot(np.arange(len(rank_y3)), rank2_y3, 'b:', label='SMBO_smac1')
#     # plt.plot(np.arange(len(rank_y1)), rank_y1, 'r', label='tf')
#     # plt.plot(np.arange(len(rank_y2)), rank_y2, 'g', label='RS')
#     # plt.plot(np.arange(len(rank_y3)), rank_y3, 'b', label='SMBO_smac')
#     # plt.title(names)
#     # plt.legend()
#     # plt.show()
#
#     # print(len(y_star), len(rank_y1))
# #     # plotting the curves
#     matrx21[:, count] = y1[:200]
#     print(count)
#     count += 1
# #     matrx22[:, count] = rank_y2[:40]
# #     matrx23[:, count] = rank_y3[:40]
# #     matrx31[:, count] = rank2_y1[:40]
# #     matrx32[:, count] = rank2_y2[:40]
# #     matrx33[:, count] = rank2_y3[:40]
# try:
#     while matrx21[0, count]==0:
#         matrx21 = np.delete(matrx21, count, 1)
# except:
#     pass
#
# p1 = np.average(matrx21, axis=1)
# # p2 = np.average(matrx22, axis=1)
# # p3 = np.average(matrx23, axis=1)
# # q1 = np.average(matrx31, axis=1)
# # q2 = np.average(matrx32, axis=1)
# # q3 = np.average(matrx33, axis=1)
# # p3 = np.average(matrx23, axis=1)
# # p1 = np.append(0, p1)
# # p3 = np.append(0, p3)
# # p1 = p1[1:]
# # p2 = p2[1:]
# # p3 = p3[1:]
# # print(p1, p2, p3)
# m01 = np.arange(len(p1))
# # m02 = np.arange(len(p2))
# # m03 = np.arange(len(p3))
# plt.plot(m01, p1, 'r--', label='SMAC')
# # plt.plot(m02, p3, 'b-.', label='smbo_smac')
# # plt.plot(m02, p1, 'g:', label='smbotf')
# # plt.yscale('log')
# # plt.plot(m01, q2, 'r', label='Random Sampling')
# # plt.plot(m02, q3, 'b', label='smbo')
# # plt.plot(m02, q1, 'g', label='smbotf')
# plt.legend()
# plt.show()
# # plt.savefig('/home/anand7s/Desktop/Transfer_hpo_l/RS_benchmarkTable/RS_vs_bohb_plot_averaged_N.png')