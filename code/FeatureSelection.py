import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# import MedianAE
# from MLPModel import MyCLModel
# from utils import cal_bin_idx, cal_centr, cal_alpha, set_seed, add_gaussian_noise, cal_centr_by_train, cal_sort_point, cal_uniformed_sort_point, cal_fused_sort_point
# from CLLoss import Contrastive_Loss
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor # 导入随机森林分类器和回归器

# from MedianAE import median_absolute_error
import csv
import os

if __name__ == "__main__":
    path = './parkinsons_updrs.data'
    data = pd.read_csv(path, sep=',')
    header = data.columns.tolist()[6:]
    for i in range(len(header)):
        header[i] = header[i].replace("Jitter", "Jitter\n").replace("Shimmer", "Shimmer\n")

    data = data.values

    X = data[:, 6:]
    y = data[:, 4:6]

    # print("X={}".format(X))
    # print("y={}".format(y))

    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=2024)

    ss = StandardScaler()

    train_data = np.concatenate((train_X, train_Y), axis=1)
    test_data = np.concatenate((test_X, test_Y), axis=1)
    train_data = ss.fit_transform(train_data)
    test_data = ss.transform(test_data)

    train_X = train_data[:, 0:16]
    train_Y = train_data[:, 16:]
    test_X = test_data[:, 0:16]
    test_Y = test_data[:, 16:]

    random_seeds = [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033]
    # random_seeds = [2024]

    feature_importances_all = []

    for seed in random_seeds:
        # 初始化随机森林模型
        model = RandomForestRegressor(random_state=seed)

        # 训练模型
        model.fit(train_X, train_Y[:, 0])

        # 获取特征重要性
        feature_importances = model.feature_importances_

        # 将特征重要性添加到列表中
        feature_importances_all.append(feature_importances)

    feature_importances_all = np.array(feature_importances_all)
    # print(feature_importances_all.shape)
    feature_importances = np.mean(feature_importances_all, axis=0)

    # 使用 zip 函数将特征列表 X 和标签列表 y 组合成元组列表
    combined = zip(feature_importances, header)

    # 根据特征列表 X 的值对元组列表进行排序
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # 使用列表解析将排序后的标签列表提取出来
    sorted_header = [item[1] for item in sorted_combined]
    sorted_importances = [item[0] for item in sorted_combined]

    plt.figure(figsize=(10, 8))
    plt.bar(sorted_header, sorted_importances)
    plt.title('RF Feature Selection For Motor UPDRS', fontsize=24)
    plt.xlabel('Features', fontsize=24)
    plt.ylabel('Importances', fontsize=24)
    plt.xticks(fontsize=12)  # 设置X轴标签字体大小为12
    plt.yticks(fontsize=12)

    # 显示图形
    plt.savefig('Motor.png')