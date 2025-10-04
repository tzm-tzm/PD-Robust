import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import MedianAE
from MLPModel import MyCLModel
from utils import cal_bin_idx, cal_centr, cal_alpha, set_seed, add_gaussian_noise, cal_centr_by_train, cal_sort_point, cal_uniformed_sort_point, cal_fused_sort_point
from CLLoss import Contrastive_Loss
from normalization import zscore_normalization1, MinMaxNormalization, zscore_normalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, PoissonRegressor, GammaRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入决策树分类器和回归器
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor, StackingRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import TruncatedSVD

from MedianAE import median_absolute_error
from scipy.stats import ttest_rel
import sys
import csv
import os

import anfis, membership, experimental
from minisom import MiniSom
import argparse

def TrainSOM(train_X):
    ## 训练SOM
    som = MiniSom(3, 3, train_X.shape[1], sigma=0.5, learning_rate=0.5, neighborhood_function='gaussian', random_seed=2024)
    # som.pca_weights_init(train_X)
    som.train_batch(train_X, 1000, verbose=False)

    som_shape = (3, 3)
    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in train_X]).T

    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    return som, cluster_index

def TrainSVD(train_X, k, cluster_index):
    # Train SVD for each cluster
    SVD_list = []
    SVD_X = np.zeros((train_X.shape[0], k), dtype=float)
    # ANFIS_list = []

    for c in np.unique(cluster_index):
        svd = TruncatedSVD(n_components=k)
        SVD_X[cluster_index == c, :] = svd.fit_transform(train_X[cluster_index == c, :])
        SVD_list.append(svd)
    return SVD_list, SVD_X

## np.array
def TrainTestANFIS(SVD_X, test_X, train_Y, test_Y, k, cluster_index, SVD_list, j, show_plots=False, device="cuda:0"):
    ## 训练SOM
    # som, cluster_index = TrainSOM(train_X)

    # SVD_list, SVD_X = TrainSVD(train_X, k, cluster_index)
    
    train_index = cluster_index
    SVD_X = torch.tensor(SVD_X, dtype=torch.float)
    # train_X = torch.tensor(train_X, dtype=torch.float)
    # test_X = torch.tensor(test_X, dtype=torch.float)
    train_Y = torch.tensor(train_Y, dtype=torch.float)
    # test_Y = torch.tensor(test_Y, dtype=torch.float)

    # Train ANFIS for each cluster
    for c in np.unique(train_index):
        if (k==3):
            invardefs = [
                    ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                    ('X2', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                    ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696]))
            ]
        elif (k==4):
            invardefs = [
                    ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                    ('X2', membership.make_bell_mfs(0.444045, 2,    [0.425606, 0, 1.313696])),
                    ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                    ('X4', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696]))
                    ]
        elif (k==5):
            invardefs = [
                    ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                    ('X2', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                    ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                    ('X4', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                    ('X5', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696]))
                    # ('X6', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696]))
                    ]
        elif (k==6):
            invardefs = [
                    ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                    ('X2', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                    ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                    ('X4', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                    ('X5', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                    ('X6', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696]))
                    ]
        elif (k==7):
            invardefs = [
                    ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                    ('X2', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                    ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                    ('X4', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                    ('X5', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                    ('X6', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                    ('X7', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696]))
                    ]
        elif (k==8):
            invardefs = [
                    ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                    ('X2', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                    ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                    ('X4', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                    ('X5', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                    ('X6', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                    ('X7', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                    ('X8', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696]))
                    ]
        outvars = ['Motor']

        model = anfis.AnfisNet('Cluster {}'.format(c),  invardefs, outvars)
        model = model.float()

        train_set = DataLoader(TensorDataset(SVD_X[train_index ==  c, :], train_Y[train_index ==  c, j]), batch_size=int(np.sum(train_index == c)), shuffle=True)
        experimental.train_anfis(model, train_set, 500, show_plots=show_plots, cluster=c, device=device)

    # Test
    # test_set = DataLoader(TensorDataset(test_X, test_Y), batch_size=test_X.size()[0], shuffle=True)

    ## ANFIS Inference
    all_y_preds = []
    test_num = len(test_X)

    i = 0
    for c in np.unique(train_index):
    # for c in [0]:
        # test_set = DataLoader(TensorDataset(test_X, test_Y), batch_size=test_X.size()[0], shuffle=True)
        test_model = torch.load('./ANFIS{}/ANFIS{}.pth'. format(int(k), int(c)))
        svd = SVD_list[i]
        i += 1
        test_X_SVD = svd.transform(test_X)
        test_set = DataLoader(TensorDataset(torch.tensor(test_X_SVD, dtype=torch.float), torch.tensor(test_Y[:, j], dtype=torch.float)), batch_size=test_num, shuffle=True)
        y_pred = experimental.test_anfis(test_model, test_set, show_plots=show_plots)
        all_y_preds.append(y_pred)
    all_y_preds = torch.stack(all_y_preds)
    mean_y_pred = torch.mean(all_y_preds, dim=0)

    # Evaluation
    rmse, mae, medae = experimental.calc_error(mean_y_pred, torch.tensor(test_Y[:, j]))
    print('RMSE={:.5f}, MAE={:.5f}, MedAE={:.5f}'
          .format(rmse, mae, medae))
    return rmse, mae, medae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k", type=int, help="SVD_k")
    # parser.add_argument("bin_num", type=int, help="bin_num")

    args = parser.parse_args()
    k = args.k
    # bin_num = args.bin_num
    init_k = k

    set_seed(2024)
    # temp_list = [0.07, 0.2, 0.3, 1, 2, 5, 10]
    # bin_num_list = [3,7,9,11,13,15,17,20,25,30,35]
    bin_num_list = [5]
    # bin_num_list = [5]
    # noise_list = [40, 30, 20, 10]
    # noise_list = [5, 15, 25, 35]
    # noise_list = [5, 10, 15, 20, 25, 30, 35]
    noise_list = [20]
    # random_seeds = [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033]
    random_seeds = [2024, 2025, 2026]

    # if not os.path.exists('./bin' + str(bin_num)):
    #     os.mkdir('./bin' + str(bin_num))

    for noise in noise_list:
        for bin_num in bin_num_list:
            for j in range(2):
                all_results = np.zeros((len(random_seeds), 2, 3))
                for seed_idx in range(len(random_seeds)):
                    k = init_k
                    np.random.seed(random_seeds[seed_idx])
                    path = './parkinsons_updrs.data'
                    data = pd.read_csv(path, sep=',')
                    data = data.values

                    X = data[:, 6:]
                    y = data[:, 4:6]

                    # print("X={}".format(X))
                    # print("y={}".format(y))

                    train_X, test_X, train_Y, test_Y =train_test_split(X, y, test_size=0.4893,random_state=2024)
                    sort_idx = train_X[:, 14]

                    # print("sort_idx={}".format(sort_idx))

                    ### 均匀分布 ###
                    sort_point = cal_uniformed_sort_point(bin_num, sort_idx)
                    # print("sort point={}".format(sort_point))

                    # root_folder_path = './bin' + str(bin_num) + '/' + str(noise) + 'dB'
                    # if not os.path.exists('./bin' + str(bin_num) + '_all_ensemble'):
                    #     os.mkdir('./bin' + str(bin_num) + '_all_ensemble')
                    # root_folder_path = './bin' + str(bin_num) + '_all_ensemble/' + str(noise) + 'dB'
                    # if not os.path.exists(root_folder_path):
                    #     os.mkdir(root_folder_path)

                    root_folder_path = f'../experiment_3.5/JRAP/{noise}dB/'
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

                    model = torch.load('../experiment_3.5/bin_num/bin' + str(bin_num) +'/JRAP_Res_MLP.pth')
                    device = model.device

                    train_bin_idx = torch.tensor(train_X[:, 0], dtype=torch.int)
                    sample_embed = torch.tensor(train_X[:, 1:], dtype=torch.float)
                    train_alpha = cal_alpha(bin_num, train_bin_idx)
                    train_alpha = train_alpha.to(device)
                    train_bin_idx = train_bin_idx.to(device)
                    sample_embed = sample_embed.to(device)
                    test_bin_idx = test_bin_idx.to(device)
                    test_X_norm = test_X.to(device)

                    model = model.to(device)
                    model.eval()
                    train_h, _, train_bin_centr = model(sample_embed, train_bin_idx, train_alpha, bin_num)
                    test_sample_centr = cal_centr_by_train(test_bin_idx, train_bin_centr)
                    train_bin_centr = train_bin_centr.to(device)
                    test_sample_centr = test_sample_centr.to(device)
                    h, _, _ = model(test_X_norm, test_bin_idx, test_alpha, bin_num)

                    train_h = train_h.detach().cpu().numpy()
                    test_h = h.detach().cpu().numpy()
                    train_X_cat = np.concatenate((train_X[:, 1:], train_h), axis=1)
                    test_X_cat = np.concatenate((test_X, test_h), axis=1)

                    _, cluster_index_X = TrainSOM(train_X[:, 1:])
                    _, cluster_index_h = TrainSOM(train_h)
                    # _, cluster_index_cat = TrainSOM(train_X_cat)

                    # cluster_index_test_X = []
                    # cluster_index_test_h = []

                    # for i in range(test_X.shape[0]):
                    #     bmu = som_X.winner(test_X[i])  # 获取BMU的坐标
                    #     cluster_index = bmu[0] * som_X.y + bmu[1]  # 将二维坐标转为一维索引
                    #     cluster_index_test_X.append(cluster_index)

                    #     bmu = som_h.winner(test_h[i])
                    #     cluster_index = bmu[0] * som_h.y + bmu[1]  # 将二维坐标转为一维索引
                    #     cluster_index_test_h.append(cluster_index)

                    # cluster_index_test_X = np.array(cluster_index_test_X)
                    # cluster_index_test_h = np.array(cluster_index_test_h)

                    SVD_list_X, SVD_train_X = TrainSVD(train_X[:, 1:], k, cluster_index_X)
                    SVD_list_h, SVD_train_h = TrainSVD(train_h, k, cluster_index_h)
                    # SVD_list_cat, SVD_train_cat = TrainSVD(train_X_cat, 2*k, cluster_index_cat)
                    # print(list(c for c in np.unique(cluster_index_X)))
                    # print(list(c for c in np.unique(cluster_index_h)))
                    # SVD_test_X = np.zeros_like(test_X)
                    # SVD_test_h = np.zeros_like(test_h)

                    # for c in np.unique(cluster_index_X):
                    #     svd_X, svd_h = SVD_list_X[c], SVD_list_h[c]
                    #     SVD_test_X[cluster_index_test_X==c] = svd_X.transform(test_X[cluster_index_test_X==c])
                    #     SVD_test_h[cluster_index_test_h==c] = svd_h.transform(test_h[cluster_index_test_h==c])

                    train_X_cat = np.concatenate((SVD_train_X, SVD_train_h), axis=1)
                    # print(train_X_cat.shape)
                    # train_h_cat = np.concatenate((SVD_test_h, SVD_test_X), axis=1)

                    show_plots = False

                    method_name = ["X", "single_h", "cat"]
                    err_name = ["RMSE", "MAE", "MedianAE"]
                    y_name = ["Motor", "Total"]

                    err = np.zeros((2, len(err_name)))
                    for i in range(2):
                        if(i==0):
                            err[i, :] = TrainTestANFIS(SVD_train_X, test_X, train_Y, test_Y, k, cluster_index_X, SVD_list_X, j, show_plots=show_plots, device=device)
                        # elif(i==1):
                        #     err[i, :] = TrainTestANFIS(SVD_train_h, test_h, train_Y, test_Y, k, cluster_index_h, SVD_list_h, j, show_plots=show_plots, device=device)
                        elif(i==1):
                            k = 2*init_k
                            # err[i, :] = TrainTestANFIS(SVD_train_cat, test_X_cat, train_Y, test_Y, 2*k, cluster_index_cat, SVD_list_cat, j, show_plots=show_plots, device=device)
                            train_X_cat = torch.tensor(train_X_cat, dtype=torch.float)
                            train_Y = torch.tensor(train_Y, dtype=torch.float)

                            # Train ANFIS for each cluster
                            for c in np.unique(cluster_index_X):
                                if (k==3):
                                    invardefs = [
                                            ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                                            ('X2', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                                            ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696]))
                                    ]
                                elif (k==4):
                                    invardefs = [
                                            ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                                            ('X2', membership.make_bell_mfs(0.444045, 2,    [0.425606, 0, 1.313696])),
                                            ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                                            ('X4', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696]))
                                            ]
                                elif (k==5):
                                    invardefs = [
                                            ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                                            ('X2', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                                            ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                                            ('X4', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                                            ('X5', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696]))
                                            # ('X6', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696]))
                                            ]
                                elif (k==6):
                                    invardefs = [
                                            ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                                            ('X2', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                                            ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                                            ('X4', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                                            ('X5', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                                            ('X6', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696]))
                                            ]
                                elif (k==7):
                                    invardefs = [
                                            ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                                            ('X2', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                                            ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                                            ('X4', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                                            ('X5', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                                            ('X6', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                                            ('X7', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696]))
                                            ]
                                elif (k==8):
                                    invardefs = [
                                            ('X1', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                                            ('X2', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                                            ('X3', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                                            ('X4', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                                            ('X5', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696])),
                                            ('X6', membership.make_tri_mfs(1.0, [0.425606,  0, 1.313696])),
                                            ('X7', membership.make_bell_mfs(0.444045, 3,    [0.425606, 0, 1.313696])),
                                            ('X8', membership.make_gauss_mfs(1.0, [0.425606, 0, 1.313696]))
                                            ]
                                outvars = [y_name[j]]

                                model = anfis.AnfisNet('Cluster {}'.format(c),  invardefs, outvars)
                                model = model.float()

                                train_set = DataLoader(TensorDataset(train_X_cat[cluster_index_X ==  c, :], train_Y[cluster_index_X ==  c, j]), batch_size=int(np.sum(cluster_index_X == c)),       shuffle=True)
                                experimental.train_anfis(model, train_set, 500, show_plots=show_plots, cluster=c, device=device)

                            # Test
                            # test_set = DataLoader(TensorDataset(test_X, test_Y), batch_size=test_X.size()[0], shuffle=True)

                            ## ANFIS Inference
                            all_y_preds = []
                            test_num = len(test_X)

                            buffer = 0
                            # buffer_X = 0
                            # buffer_h = 0
                            for c in np.unique(cluster_index_X):
                                test_model = torch.load('./ANFIS{}/ANFIS{}.pth'. format(int(k), int(c)))
                                svd_X = SVD_list_X[c]
                                if c in cluster_index_h:
                                    svd_h = SVD_list_h[buffer]
                                    buffer += 1
                                else:
                                    svd_h = SVD_list_h[buffer]
                                # buffer_h = 0
                                # for _ in np.unique(cluster_index_h):
                            # for c in [0]:
                                # test_set = DataLoader(TensorDataset(test_X, test_Y), batch_size=test_X.size()[0], shuffle=True)
                                # if c+1 in cluster_index_h:
                                #     buffer_h += 1
                                test_X_SVD = np.concatenate((svd_X.transform(test_X), svd_h.transform(test_h)), axis=1)
                                test_set = DataLoader(TensorDataset(torch.tensor(test_X_SVD, dtype=torch.float), torch.tensor(test_Y[:, j], dtype=torch.float)), batch_size=test_num, shuffle=True)
                                y_pred = experimental.test_anfis(test_model, test_set, show_plots=show_plots)
                                all_y_preds.append(y_pred)
                                # buffer_X += 1
                                # buffer += 1
                            all_y_preds = torch.stack(all_y_preds)
                            mean_y_pred = torch.mean(all_y_preds, dim=0)

                            # Evaluation
                            err[i, :] = experimental.calc_error(mean_y_pred, torch.tensor(test_Y[:, j]))
                            print('RMSE={:.5f}, MAE={:.5f}, MedAE={:.5f}'
                                  .format(err[i, 0], err[i, 1], err[i, 2]))
                    all_results[seed_idx, :, :] = err

                # p = np.zeros(len(err_name))

                # for k in range(len(err_name)):
                #     _, p_value = ttest_rel(all_results[:, 0, k], all_results[:, 1, k])
                #     p[k] = p_value

                mean_of_all = np.mean(all_results, axis=0)
                variance_of_all = np.std(all_results, ddof=1, axis=0)            
                folder_path = root_folder_path

                file_name = y_name[j] + '_mean.csv'
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'a', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerows(['{:.3f}'.format(value) for value in row] for row in mean_of_all)
                file_name = y_name[j] + '_std.csv'
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'a', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerows(['{:.3f}'.format(value) for value in row] for row in variance_of_all)
                # file_name = y_name[j] + '_p.csv'
                # file_path = os.path.join(folder_path, file_name)
                # with open(file_path, 'a', newline='') as file:
                #     csv_writer = csv.writer(file)
                #     csv_writer.writerows(['{:.3f}'.format(value)] for value in p)

if __name__ == "__main__":
    main()