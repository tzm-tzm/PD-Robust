import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import MedianAE
from MLPModel import MyCLModel
from utils import cal_bin_idx, cal_centr, cal_alpha, set_seed, add_gaussian_noise, cal_centr_by_train, cal_sort_point, cal_uniformed_sort_point, cal_fused_sort_point
from CLLoss import Contrastive_Loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.linear_model import LinearRegression

from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入决策树分类器和回归器
from sklearn.ensemble import BaggingClassifier, BaggingRegressor  # 导入Bagging分类器和回归器
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor  # 导入AdaBoost分类器和回归器
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # 导入随机森林分类器和回归器

from MedianAE import median_absolute_error
import csv
import os

if __name__ == "__main__":
    # err_num = 2
    # noise = 40
    # method = 2
    method_list = [1,2]
    noise_list = [5, 10, 15, 20, 25, 30, 35]
    lower_lim_list = np.array([[-1, -1, -1],
                              [-0.1, -0.1, -0.1]])
    upper_lim_list = np.array([[0.5, 0.5, 0.5],
                              [0.025, 0.025, 0.025]])
    for method in method_list:
        for err_idx in [0, 1, 2]:
            method_name = ['Augmented', 'Aug Combined']
            err_name = ['RMSE', 'MAE', 'MedianAE']
            downstream_name = ['MLR', 'SVR', 'NN$_1$', 'NN$_2$', 'AdaBoost', 'Bagging', 'GPR']
            # downstream_name = ['SVR', 'NN1', 'NN2', 'AdaBoost', 'Bagging', 'GPR']
            downstream_num = len(downstream_name)
            all_err = np.zeros((len(method_list), len(noise_list), downstream_num, 3))
            all_std = np.zeros((len(method_list), len(noise_list), downstream_num, 3))
            for k in range(len(noise_list)):
                err = np.zeros((downstream_num, 3))
                std_e = np.zeros((downstream_num, 3))
                folder_path = './' + str(noise_list[k]) + 'dB'
                # if not os.path.exists(folder_path):
                #     os.mkdir(folder_path)
                mean = np.genfromtxt(folder_path + '/Motor_mean.csv', delimiter=',')
                std = np.genfromtxt(folder_path +  '/Motor_std.csv', delimiter=',')
                # cov = np.genfromtxt(folder_path + '/total_cov.csv', delimiter=',')
                for j in range(1, downstream_num):
                # for j in [2,3,4,5]:
                    mean_B = mean[3 * j, :]
                    mean_A = mean[3 * j + method, :]
                    std_B = std[3 * j, :]
                    std_A = std[3 * j + method, :]
                    err[j, :] = (mean_A - mean_B) / mean_B
                    # std_e[j, :] = np.sqrt(((mean_A/mean_B)**2)*(((std_A/mean_A)**2)+((std_B/mean_B)**2)-2*cov[j, :]/(mean_A*mean_B)))
                    std_e[j, :] = np.sqrt(((mean_A / mean_B) ** 2) * (
                            ((std_A / mean_A) ** 2) + ((std_B / mean_B) ** 2)))
                all_err[method-1, k, :, :] = err
                all_std[method-1, k, :, :] = std_e

            plt.figure(figsize=(16,12))
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
            for i in range(1, downstream_num):
                # plt.plot(binnum_list, all_err[:, i, 0], marker='o', linestyle='-', color=colors[i], label=downstream_name[i])
                plt.errorbar(noise_list, all_err[method-1, :, i, err_idx], yerr=all_std[method-1, :, i, err_idx], fmt='o', linestyle='-', color=colors  [i], label=downstream_name[i])
                plt.scatter(noise_list, all_err[method-1, :, i,  err_idx]+all_std[method-1, :, i, err_idx], marker='_', color=colors[i])  # 绘制下端点
                plt.scatter(noise_list, all_err[method-1, :, i,  err_idx]-all_std[method-1, :, i, err_idx], marker='_', color=colors[i])  # 绘制上端点

            plt.axhline(y=0, color='black', linestyle='--')
            plt.legend(fontsize=24, loc='lower left')
            plt.title(err_name[err_idx]+' Relative Error of {}'.format(method_name[method-1]), fontsize=48)
            plt.xlabel('Noise(dB)', fontsize=32)
            plt.ylabel('Relative Error', fontsize=32)
            plt.xticks(fontsize=32) 
            plt.yticks(fontsize=32)
            plt.ylim(lower_lim_list[method-1, err_idx], upper_lim_list[method-1, err_idx])
            plt.savefig(err_name[err_idx] + '_' + method_name[method-1] + '.png')