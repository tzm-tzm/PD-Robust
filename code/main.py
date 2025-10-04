import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import MedianAE
from MLPModel import MyCLModel
from utils import cal_bin_idx, cal_centr, cal_alpha, set_seed, add_gaussian_noise, cal_centr_by_train, cal_sort_point, cal_uniformed_sort_point, cal_fused_sort_point
from CLLoss import Contrastive_Loss
# from normalization import zscore_normalization1, MinMaxNormalization, zscore_normalization
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
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bin_num", type=int, help="bin_num")

    args = parser.parse_args()
    bin_num =args.bin_num

    set_seed(2024)
    path = './parkinsons_updrs.data'
    data = pd.read_csv(path, sep=',')
    data = data.values

    X = data[:, 6:]
    y = data[:, 4:6]

    print("X={}".format(X))
    print("y={}".format(y))

    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.4893, random_state=2024)
    sort_idx = train_X[:, 14]

    print("sort_idx={}".format(sort_idx))

    ### 多bin融合 ###
    # sort_point = cal_fused_sort_point(bin_num, sort_idx)
    # bin_num = len(sort_point) - 1

    ### 每段样本量相同 ###
    # sort_point = cal_sort_point(bin_num, sort_idx)

    ### 均匀分布 ###
    sort_point = cal_uniformed_sort_point(bin_num, sort_idx)
    print("sort point={}".format(sort_point))

    if not os.path.exists('./bin_num'):
        os.mkdir('./bin_num')
    folder_path = './bin_num/bin' + str(bin_num)
    # folder_path = './' + str(noise)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    ### Add Noise ###
    # test_X = add_gaussian_noise(test_X, snr=noise)
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

    device = "cuda:1"
    temprature = 1.0
    epochs = 200
    model = MyCLModel(16, 16, loss_function=Contrastive_Loss(temp=temprature), num_layers=1, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model = model.to(device)

    # train and validation ##
    val_loss = []
    kf = KFold(n_splits=10, shuffle=False)
    for train_idx, val_idx in kf.split(train_X):
        print("train_idx={}, val_idx={}".format(train_idx, val_idx))
        train_X_fold, val_X_fold = train_X[train_idx], train_X[val_idx]

        train_bin_idx = torch.tensor(train_X_fold[:, 0], dtype=torch.int)
        val_bin_idx = torch.tensor(val_X_fold[:, 0], dtype=torch.int)

        train_alpha = cal_alpha(bin_num, train_bin_idx)
        val_alpha = cal_alpha(bin_num, val_bin_idx)

        sample_embed = torch.tensor(train_X_fold[:, 1:], dtype=torch.float)
        val_X = torch.tensor(val_X_fold[:, 1:], dtype=torch.float)

        print("train_bin_idx={}, val_bin_idx={}, sample_embed={}, val_X_norm={}".format(train_bin_idx, val_bin_idx,
                                                                                        sample_embed, val_X))
        print("train_bin_idx_size={}, val_bin_idx_size={}, sample_embed_size={}, val_X_norm_size={}".format(
            train_bin_idx.size(), val_bin_idx.size(), sample_embed.size(), val_X.size()))

        train_alpha = train_alpha.to(device)
        train_bin_idx = train_bin_idx.to(device)
        val_bin_idx = val_bin_idx.to(device)
        sample_embed = sample_embed.to(device)
        val_X = val_X.to(device)
        val_alpha = val_alpha.to(device)

        for epoch in range(epochs):
            model.train()
            h, loss, train_bin_centr = model(sample_embed, train_bin_idx, train_alpha, bin_num)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            print("epoch={},h={}, train_loss={}".format(epoch, h, loss))

            model.eval()
            stopper_count = 0
            with torch.no_grad():
                val_sample_centr = cal_centr_by_train(val_bin_idx, train_bin_centr)
                h, _, _ = model(val_X, val_bin_idx, val_alpha, bin_num)
                loss_fc = Contrastive_Loss(temp=temprature)
                train_bin_centr = train_bin_centr.to(device)
                val_sample_centr = val_sample_centr.to(device)
                loss = loss_fc(h, train_bin_centr, val_sample_centr, val_alpha)
                print("epoch={}, val_loss={}".format(epoch, loss))
                if (len(val_loss) >= 10 and loss < min(val_loss)):
                    torch.save(model, folder_path + '/Res_MLP.pth')
                if (len(val_loss) >= 10 and loss > min(val_loss[epoch - 10:])):
                    stopper_count += 1
                    if (stopper_count >= 10):
                        print("Validation loss has not decreased for 10 consecutive epochs. Stopping training.")
                        break
                else:
                    stopper_count = 0
                val_loss.append(loss)

if __name__ == "__main__":
    main()

    ### test ###
    # bin_num_list = ['3_7', '3_7_10', '3_10', '7_10']
    # bin_num_list = [2, 3, 5, 7, 10, 20, 30, 50, 100]
    # bin_num_list = [2, 3, 4, 5, 6, 7, 9, 10, 20, 50, 100]
    # bin_num_list = [5, 10, 50, 100]
    # bin_num_list = [bin_num]
    # train_bin_idx = torch.tensor(train_X[:, 0], dtype=torch.int)
    # sample_embed = torch.tensor(train_X[:, 1:], dtype=torch.float)
    # train_alpha = cal_alpha(bin_num, train_bin_idx)
    # train_alpha = train_alpha.to(device)
    # train_bin_idx = train_bin_idx.to(device)
    # sample_embed = sample_embed.to(device)
    # test_bin_idx = test_bin_idx.to(device)
    # test_X_norm = test_X.to(device)
    # for num in bin_num_list:
    #     # model = torch.load('./fused/bin_num' + str(num) + '/Res_MLP.pth')
    #     # model = torch.load('./uniformed/bin_num5/Res_MLP.pth')
    #     model = torch.load('./Res_MLP.pth')
    #     model.eval()
    #     _, _, train_bin_centr = model(sample_embed, train_bin_idx, train_alpha, bin_num)
    #     test_sample_centr = cal_centr_by_train(test_bin_idx, train_bin_centr)
    #     train_bin_centr = train_bin_centr.to(device)
    #     test_sample_centr = test_sample_centr.to(device)
    #     h, _, _ = model(test_X_norm, test_bin_idx, test_alpha, bin_num)
    #     loss_fc = Contrastive_Loss(temp=temprature)
    #     loss = loss_fc(h, train_bin_centr, test_sample_centr, test_alpha)
    #     print("Test: h={}, loss={}".format(h, loss))

        ## Downstream model ##
        # train_numbers = train_bin_idx.detach().cpu().numpy()
        # train_h, _, _ = model(sample_embed, train_bin_idx, train_alpha, bin_num)
        # train_h = train_h.detach().cpu().numpy()
        # train_X_cat = np.concatenate((train_h, train_X[:, 1:]), axis=1)
        # train_X_mean = np.add(train_h, train_X[:, 1:])/2
        #
        # test_size = test_Y.shape[0]
        # test_numbers = test_bin_idx.detach().cpu().numpy()
        # test_h = h.detach().cpu().numpy()
        # test_X_cat = np.concatenate((test_h, test_X), axis=1)
        # test_X_mean = np.add(test_h, test_X) / 2
        #
        # downstream_model_num = 6
        # method_name = [ "single_X", "single_h", "cat", "mean"]
        # err_name = ["RMSE", "MAE", "MedianAE", "R2_Score"]
        # y_name = ["Motor", "Total"]
        # for k in range(len(method_name)):
        #     for j in range(len(y_name)):
        #         err = np.zeros((downstream_model_num, len(err_name)))
        #         for m in range(downstream_model_num):
        #             predict_model = []
        #             # Training #
        #             if(m==0):
        #                 predict_model = LinearRegression(fit_intercept=True)
        #             elif(m==1):
        #                 predict_model = svm.SVR(kernel='linear')
        #             elif(m==2):
        #                 predict_model = svm.SVR(kernel='poly')
        #             elif(m==3):
        #                 predict_model = svm.SVR(kernel='rbf', gamma=1, C=4)
        #             elif(m==4):
        #                 predict_model = MLPRegressor(solver="sgd", alpha=1e-3, activation="relu", hidden_layer_sizes=(32, 64, 32),
        #                                                                        max_iter=2000, tol=1e-3, random_state=2024, verbose=False)
        #             elif(m==5):
        #                 predict_model = DecisionTreeRegressor(random_state=2024)
        #
        #             if(k == 0):
        #                 predict_model.fit(train_X[:, 1:], train_Y[:, j])
        #             elif(k==1):
        #                 predict_model.fit(train_h, train_Y[:, j])
        #             elif(k==2):
        #                 predict_model.fit(train_X_cat, train_Y[:, j])
        #             elif(k==3):
        #                 predict_model.fit(train_X_mean, train_Y[:, j])
        #
        #             # Testing #
        #             if(k==0):
        #                 y_pred = predict_model.predict(test_X)
        #             elif(k==1):
        #                 y_pred = predict_model.predict(test_h)
        #             elif(k==2):
        #                 y_pred = predict_model.predict(test_X_cat)
        #             elif(k==3):
        #                 y_pred = predict_model.predict(test_X_mean)
        #
        #             y_true = test_Y[:, j]
        #
        #             MSE = np.mean((y_pred - y_true)**2)
        #             MAE = np.sum(abs(y_pred - y_true))
        #             RMSE = np.sqrt(MSE)
        #             MAE = MAE / test_size
        #             MedAE = median_absolute_error(y_true, y_pred)
        #             VAR = y_true.var()
        #             r_2 = 1 - (MSE/VAR)
        #             err[m, 0] = RMSE
        #             err[m, 1] = MAE
        #             err[m, 2] = MedAE
        #             err[m, 3] = r_2
        #             print("RMSE={}, MAE={}, MedianAE={}, r2_score={}".format(RMSE, MAE, MedAE, r_2))
        #             np.savetxt(y_name[j] + '_' + method_name[k] + '.csv', err, delimiter=',', fmt="%.4f", header='RMSE,MAE,MedianAE,R2',
        #                        comments='')
        #
        # all_data = []
        # err_name = ["Motor_single_X.csv", "Total_single_X.csv", "Motor_single_h.csv", "Total_single_h.csv",
        #             "Motor_cat.csv", "Total_cat.csv", "Motor_mean.csv", "Total_mean.csv"]
        # # 遍历10个CSV文件
        # for i in range(8):
        #     filename = err_name[i]  # CSV文件名格式
        #     if os.path.isfile(filename):  # 确保文件存在
        #         with open(filename, 'r') as file:
        #             csv_reader = csv.reader(file)
        #             # 将当前文件的数据添加到all_data列表中
        #             all_data.extend(list(csv_reader))
        #
        #
        # # 将所有数据写入新的CSV文件
        # file_name = "merged_data" + str(num) + ".csv"
        # file_path = os.path.join(folder_path, file_name)
        # with open(file_path, 'w', newline='') as file:
        #     csv_writer = csv.writer(file)
        #     csv_writer.writerows(all_data)