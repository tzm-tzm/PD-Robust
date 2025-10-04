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
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, PoissonRegressor, GammaRegressor

from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入决策树分类器和回归器
from sklearn.ensemble import BaggingClassifier, BaggingRegressor  # 导入Bagging分类器和回归器
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor  # 导入AdaBoost分类器和回归器
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # 导入随机森林分类器和回归器
from sklearn.gaussian_process import GaussianProcessRegressor

import lightgbm as lgb

from MedianAE import median_absolute_error
from scipy.stats import ttest_rel
import csv
import os

if __name__ == "__main__":
    downstream_model_num = 4
    set_seed(2024)
    # temp_list = [0.07, 0.2, 0.3, 1, 2, 5, 10]
    # bin_num_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # bin_num_list = [5]
    # noise_list = [40, 30, 20, 10]
    # noise_list = [5, 15, 25, 35]
    noise_list = [5, 10, 15, 20, 25, 30, 35]
    # noise_list = [10, 20, 30]
    random_seeds = [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033]
    # random_seeds = [2024]
    for noise in noise_list:
        # for num in bin_num_list:
        # for temp in temp_list:
        for j in range(2):
            all_results = np.zeros((len(random_seeds), 2*downstream_model_num, 3))
            for seed_idx in range(len(random_seeds)):
                np.random.seed(random_seeds[seed_idx])
 
                path = './parkinsons_updrs.data'
                data = pd.read_csv(path, sep=',')
                data = data.values
 
                bin_num = 5
 
                X = data[:, 6:]
                y = data[:, 4:6]
 
                print("X={}".format(X))
                print("y={}".format(y))
 
                train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.4893, random_state=2024)
                sort_idx = train_X[:, 14]
 
                print("sort_idx={}".format(sort_idx))
 
                ### 均匀分布 ###
                sort_point = cal_uniformed_sort_point(bin_num, sort_idx)
                print("sort point={}".format(sort_point))
 
                root_folder_path = './' + str(noise) + 'dB'
                if not os.path.exists(root_folder_path):
                    os.mkdir(root_folder_path)
 
                ### Add Noise ###
                test_X = add_gaussian_noise(test_X, snr=noise)
                #################
 
                train_bin_idx = cal_bin_idx(train_X[:, 14], sort_point)
                train_bin_idx = np.expand_dims(train_bin_idx, axis=1)
                test_bin_idx = cal_bin_idx(test_X[:, 14], sort_point)
                test_bin_idx = np.expand_dims(test_bin_idx, axis=1)
 
                train_X = np.concatenate((train_bin_idx, train_X), axis=1)
                test_X = np.concatenate((test_bin_idx, test_X), axis=1)
 
                train_data = np.concatenate((train_X, train_Y), axis=1)
                test_data = np.concatenate((test_X, test_Y), axis=1)
 
                ss = StandardScaler()
                train_data[:, 1:] = ss.fit_transform(train_data[:, 1:])
                test_data[:, 1:] = ss.transform(test_data[:, 1:])
 
                train_X = train_data[:, 0:17]
                train_Y = train_data[:, 17:]
                test_X = test_data[:, 0:17]
                test_Y = test_data[:, 17:]
 
                test_bin_idx = torch.tensor(test_X[:, 0], dtype=torch.int)
                test_alpha = cal_alpha(bin_num, test_bin_idx)
                test_X = torch.tensor(test_X[:, 1:], dtype=torch.float)
 
                # temprature = 1.0
                # epochs = 20
                # model = MyCLModel(16, 16, loss_function=Contrastive_Loss(temp=temprature), num_layers=1)
                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                model = torch.load('./Res_MLP.pth')
                device = model.device
                # model = model.to(device)
 
                ### test ###
                # bin_num_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                train_bin_idx = torch.tensor(train_X[:, 0], dtype=torch.int)
                sample_embed = torch.tensor(train_X[:, 1:], dtype=torch.float)
                train_alpha = cal_alpha(bin_num, train_bin_idx)
                train_alpha = train_alpha.to(device)
                train_bin_idx = train_bin_idx.to(device)
                sample_embed = sample_embed.to(device)
                test_bin_idx = test_bin_idx.to(device)
                test_X_norm = test_X.to(device)
 
                # model = torch.load('./uniformed2/bin_num5/Res_MLP.pth')
                # model = torch.load('./Res_MLP.pth')
                model.eval()
                train_h, _, train_bin_centr = model(sample_embed, train_bin_idx, train_alpha, bin_num)
                test_sample_centr = cal_centr_by_train(test_bin_idx, train_bin_centr)
                train_bin_centr = train_bin_centr.to(device)
                test_sample_centr = test_sample_centr.to(device)
                h, _, _ = model(test_X_norm, test_bin_idx, test_alpha, bin_num)
                # loss_fc = Contrastive_Loss(temp=1.0)
                # loss = loss_fc(h, train_bin_centr, test_sample_centr, test_alpha)
                # print("Test: h={}, loss={}".format(h, loss))
 
                ## Downstream model ##
                train_numbers = train_bin_idx.detach().cpu().numpy()
                train_h, _, _ = model(sample_embed, train_bin_idx, train_alpha, bin_num)
                train_h = train_h.detach().cpu().numpy()
                train_X_cat = np.concatenate((train_h, train_X[:, 1:]), axis=1)
                train_X_mean = np.add(train_h, train_X[:, 1:])/2
 
                test_size = test_Y.shape[0]
                test_numbers = test_bin_idx.detach().cpu().numpy()
                test_h = h.detach().cpu().numpy()
                test_X_cat = np.concatenate((test_h, test_X), axis=1)
                test_X_mean = np.add(test_h, test_X) / 2
 
                method_name = ["single_X", "single_h", "cat", "mean"]
                err_name = ["RMSE", "MAE", "MedianAE"]
                y_name = ["Motor", "Total"]
                # for k in range(len(method_name)):
                #     for j in range(len(y_name)):
                err = np.zeros((2*downstream_model_num, len(err_name)))

                for m in range(downstream_model_num):
                    for k in range(2):
                        predict_model = []
                        # Training #
                        if (m==0):
                            predict_model = svm.SVR(kernel='poly')
                        elif (m==1):
                            predict_model = MLPRegressor(solver="sgd", alpha=1e-3, activation="relu", hidden_layer_sizes=(32),  max_iter=2000, tol=1e-3, random_state=2024)
                        elif (m == 2):
                            predict_model = GaussianProcessRegressor()
                        elif (m == 3):
                           predict_model = BaggingRegressor(random_state=2024)
                        # elif (m==4):
                        #     params = {
                        #                 'objective': 'regression',       # 回归任务的目标
                        #                 'metric': 'rmse',                # 评估指标
                        #                 'boosting_type': 'gbdt', 
                        #                 # 'max_depth': 6,       # 使用梯度提升决策树
                        #                 'num_leaves': 31,                # 叶节点数
                        #                 'learning_rate': 0.1,            # 学习率
                        #                 # 'bagging_fraction' : 0.8,
                        #                 # 'feature_fraction' : 0.8,
                        #                 'seed': 2024                     # 随机种子
                        #             }
                        #     train_dmat = lgb.Dataset(X_train, label=y_train)
                        #     test_dmat = lgb.Dataset(X_test, label=y_test, reference=train_dmat)
                        #     predict_model = lgb.train(params, train_dmat, num_boost_round=100)

                        # elif (m==5):
                        #     predict_model = BaggingRegressor(random_state=2024)
                        # elif (m==6):
                        #     predict_model = GaussianProcessRegressor()

                        # if (m==0):
                        #     predict_model = LinearRegression(fit_intercept=True)
                        # elif (m==1):
                        #     predict_model = svm.SVR(kernel='poly')
                        # elif (m == 2):
                        #     predict_model = MLPRegressor(solver="sgd", alpha=1e-3, activation="relu", hidden_layer_sizes=(32),  max_iter=2000, tol=1e-3, random_state=2024)
                        # elif (m == 3):
                        #    predict_model = MLPRegressor(solver="sgd", alpha=1e-3, activation="relu", hidden_layer_sizes=(32, 64,  32, 16), max_iter=2000, tol=1e-3, random_state=2024)
                        # elif (m==4):
                        #     predict_model = AdaBoostRegressor(random_state=2024)
                        # elif (m==5):
                        #     predict_model = BaggingRegressor(random_state=2024)
                        # elif (m==6):
                        #     predict_model = GaussianProcessRegressor()
                        
                        if(k == 0):
                            predict_model.fit(train_X[:, 1:], train_Y[:, j])
                        # elif(k==1):
                        #     predict_model.fit(train_h, train_Y[:, j])
                        elif(k==1):
                            predict_model.fit(train_X_cat, train_Y[:, j])
                        # elif(k==3):
                        #    predict_model.fit(train_X_mean, train_Y[:, j])

                        # Testing #
                        if(k==0):
                            y_pred = predict_model.predict(test_X)
                        # elif(k==1):
                        #     y_pred = predict_model.predict(test_h)
                        elif(k==1):
                            y_pred = predict_model.predict(test_X_cat)
                        # elif(k==3):
                        #     y_pred = predict_model.predict(test_X_mean)
 
                        y_true = test_Y[:, j]
 
                        MSE = np.mean((y_pred - y_true)**2)
                        MAE = np.sum(abs(y_pred - y_true))
                        RMSE = np.sqrt(MSE)
                        MAE = MAE / test_size
                        MedAE = median_absolute_error(y_true, y_pred)
                        VAR = y_true.var()
                        err[2*m+k, 0] = RMSE
                        err[2*m+k, 1] = MAE
                        err[2*m+k, 2] = MedAE
                        print("RMSE={}, MAE={}, MedianAE={}".format(RMSE, MAE, MedAE))
                        # np.savetxt(y_name[j] + '_' + method_name[k] + '.csv', err, delimiter=',', fmt="%.4f", header='RMSE,MAE, MedianAE',
                        #               comments='')
 
                all_results[seed_idx, :, :] = err

            p = np.zeros((downstream_model_num, len(err_name)))

            for m in range(downstream_model_num):
                for k in range(len(err_name)):
                    _, p_value = ttest_rel(all_results[:, 2*m, k], all_results[:, 2*m+1, k])
                    p[m, k] = p_value

            mean_of_all = np.mean(all_results, axis=0)
            variance_of_all = np.std(all_results, ddof=1, axis=0)
            # result = []
            # for i in range(downstream_model_num):
            #     for j in range(3):
            #         result.append("{:.4e}".format(mean_of_all[i][j]))
            # result = np.array(result).reshape(downstream_model_num, 3)
 
            folder_path = root_folder_path
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
 
            file_name = y_name[j] + '_mean.csv'
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(['{:.3f}'.format(value) for value in row] for row in mean_of_all)
            file_name = y_name[j] + '_std.csv'
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(['{:.3f}'.format(value) for value in row] for row in variance_of_all)
            file_name = y_name[j] + '_p.csv'
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(['{:.3f}'.format(value) for value in row] for row in p)