import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrastive_Loss(nn.Module):
    def __init__(self, temp):
        super(Contrastive_Loss, self).__init__()
        self.temp = temp

    def forward(self, sample_embed, bin_centr, sample_centr, alpha, device="cuda:0"):
        # device = "cuda:0"
        # normalize
        sample_embed = F.normalize(sample_embed)  # [sample_num, dim]，为所有样本Embedding的矩阵
        sample_centr = F.normalize(sample_centr)  # [sample_num, dim]，为所有样本所在类中心的矩阵
        bin_centr = F.normalize(bin_centr)  # [bin_num, dim]，为所有类中心的矩阵

        pos_similarity = torch.mul(sample_embed, sample_centr).sum(dim=1)  # [sample_num, dim] -> [sample_num, 1]
        pos_similarity = pos_similarity.to(device)
        pos_similarity = torch.exp(pos_similarity / self.temp)  # [sample_num, 1]

        total_similarity = torch.matmul(sample_embed,
                                  bin_centr.transpose(0, 1))  # [sample_num, dim]*[dim, bin_num]=[sample_num, bin_num]
        total_similarity = total_similarity.to(device)
        alpha = alpha.to(device)
        total_similarity = torch.mul(total_similarity, alpha)  # [sample_num, bin_num]
        total_similarity = torch.exp(total_similarity / self.temp).sum(dim=1)  # [sample_num, bin_num] -> [sample_num, 1]

        loss = -torch.log(pos_similarity / total_similarity).mean()  # [sample_num, 1] -> [1]
        return loss