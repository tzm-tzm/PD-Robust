import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binom
import random

def nearest_bin(idx, bin_centr, num_bin, bin_num, device="cuda:0"):
    i = idx
    j = idx
    i_valid = True
    j_valid = True
    while(True):
        i += 1
        j -= 1
        if(i<0 or i > (bin_num-1)):
            i_valid = False
        if(j<0 or j > (bin_num-1)):
            j_valid = False
        if(i_valid and j_valid):
            if(num_bin[i].item()>0 and num_bin[j].item()>0):
                p=1
                new_centr = (bin_centr[i]+(bin_centr[j]*num_bin[j].item()))/(num_bin[i].item()+num_bin[j].item())
                break
        if(i_valid):
            if(num_bin[i].item()>0):
                p=2
                new_centr = bin_centr[i]/num_bin[i].item()
                break
        if(j_valid):
            if(num_bin[j].item()>0):
                p=3
                new_centr = bin_centr[j]
                break
    new_centr = new_centr.to(device)
    # print("idx={}, i={}, j={}, method={}".format(idx, i, j, p))
    return new_centr

def cal_fused_sort_point(bin_num, sort_idx):
    sort_point = []
    max_idx = max(sort_idx)
    min_idx = min(sort_idx)
    sort_point.append(float('-inf'))
    for i in range(len(bin_num)):
        for j in range(1, bin_num[i]):
            sort_point.append(min_idx + j * (max_idx - min_idx) / bin_num[i])
    sort_point.append(float('inf'))
    sort_point = list(set(sort_point))
    sort_point.sort()
    return sort_point

def cal_uniformed_sort_point(bin_num, sort_idx):
    sort_point = []
    max_idx = max(sort_idx)
    min_idx = min(sort_idx)
    sort_point.append(float('-inf'))
    for i in range(1, bin_num):
        sort_point.append(min_idx + i * (max_idx - min_idx) / bin_num)
    sort_point.append(float('inf'))
    return sort_point

def cal_sort_point(bin_num, sort_idx):
    sort_point = []
    sample_num = sort_idx.shape[0]
    sort_idx = np.sort(sort_idx)
    num_per_bin = sample_num // bin_num
    sort_point.append(float('-inf'))
    for i in range(bin_num-1):
        sort_point.append(sort_idx[(i+1)*num_per_bin-1])
    sort_point.append(float('inf'))
    return sort_point


def add_gaussian_noise(data, snr):
    # 计算信号的功率
    signal_power = np.mean(data ** 2, axis=0)

    # 计算噪声的功率
    noise_power = signal_power / (10 ** (snr / 10))

    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)

    noisy_data = data + noise

    print(noisy_data/data)

    return noisy_data

def set_seed(seed):
    # 设置PyTorch随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 当使用多个GPU时，设置所有的GPU种子

    # 可选的，确保每次运行时结果一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_in(data, low, high)->bool:
    if(data >= low and data < high):
        return True
    else:
        return False

### 给定划bin向量bin_idx[sample_num, 1]和训练集的bin_centr[bin_num, dim]，求得其该数据集上的sample_centr[sample_num, dim]
def cal_centr_by_train(bin_idx, train_bin_centr):
    length = bin_idx.size()[0]
    dim = train_bin_centr.size()[1]
    sample_centr = torch.empty((length, dim), dtype=torch.float)
    for i in range(length):
        bin = bin_idx[i].item()
        sample_centr[i] = train_bin_centr[bin]
    return sample_centr

def cal_bin_idx(sort_idx, sort_point):
    bin_idx = np.zeros_like(sort_idx, dtype=int)
    for i in range(len(sort_idx)):
        for j in range(len(sort_point)-1):
            if (is_in(sort_idx[i], sort_point[j], sort_point[j+1])):
                bin_idx[i] = j
                break
            elif (j==len(sort_point)-2):
                raise ValueError("Your index doesn't belong to ang range.")
    return bin_idx #[sample_num, 1]，表示每个样本所在的类别

def cal_centr(bin_num, sample_embed, bin_idx, device="cuda:0"):
    # p = 0
    # device = "cuda:0"
    shape = bin_idx.size()
    length = shape[0]
    dim = bin_num
    sample_centr = torch.empty_like(sample_embed, dtype=torch.float)# [sample_num, dim]，表示每个样本所在类的中心Embedding
    bin_centr = torch.zeros((dim, sample_embed.size()[1]), dtype=torch.float) #[bin_num, dim]，表示每个类的中心Embedding
    num_bin = torch.zeros(dim, dtype=torch.int) # [bin_num, 1]
    bin_centr = bin_centr.to(device)# bin_num = bin_num.to(device)

    for i in range(length):
        bin = bin_idx[i].item()
        # print(bin_centr.device, sample_embed.device)
        bin_centr[bin] += sample_embed[i]
        num_bin[bin] += 1

    for i in range(dim):
        if(num_bin[i].item() == 0):
            # p += 1
            bin_centr[i] = nearest_bin(i, bin_centr, num_bin, bin_num, device)
            # print("corrected centr:{}".format(bin_centr[i]))
        else:
            bin_centr[i] /= num_bin[i].item()
            # print("centr:{}".format(bin_centr[i]))

    for i in range(length):
        bin = bin_idx[i].item()
        sample_centr[i] = bin_centr[bin]
    # print("There are {} bins without sample.".format(p))
    return bin_centr, sample_centr

def cal_alpha(bin_num, bin_idx):
    dim = bin_num
    alpha_1 = torch.empty((dim, dim), dtype=torch.float)
    for i in range(dim):
        for j in range(dim):
            m = i + 1
            n = j + 1
            N = 2 * max(m, (dim - m))
            alpha_1[i, j] = math.comb(N, n - m + int(N/2)) / math.comb(N, int(N/2))
    alpha = torch.empty((bin_idx.shape[0], dim), dtype=torch.float)
    for i in range(bin_idx.shape[0]):
        alpha[i] = alpha_1[bin_idx[i]]
    return alpha # [sample_num, bin_num]

# if __name__ == "__main__":
#     path = './parkinsons_updrs.data'
#     data = pd.read_csv(path, sep=',')
#     data_val = data.values
#
#     sort_idx = data_val[:, 1]
#     sort_point = [0, 55, 60, 65, 70, 75, 80, 100]
#
#     bin_idx = cal_bin_idx(sort_idx, sort_point)
#     alpha = cal_alpha(bin_idx)
#     print(alpha)
#     sample_embed = torch.tensor(np.concatenate((data_val[:, 1:4], data_val[:, 6:]), axis=1))
#     bin_idx = torch.tensor(bin_idx, dtype=torch.int)
#     bin_centr, sample_centr = cal_centr(sample_embed, bin_idx)
#     print(bin_centr, sample_centr)
#
#     epochs = 500
#     model = MyCLModel(19, 19, loss_function=Contrastive_Loss, bin_idx=bin_idx, alpha=alpha)
#
#     device = "cuda:0"
#     model.to(device)
#     alpha.to(device)
#     bin_idx.to(device)
#     sample_embed.to(device)
#
#     for epoch in range(epochs):
#         model.train()
#         h, loss = model(sample_embed, bin_idx)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         print("epoch={},loss={}".format(epoch, loss))
#
#     torch.save(model, 'my_model.pth')

