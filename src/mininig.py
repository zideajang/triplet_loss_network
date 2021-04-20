import os
import random

from itertools import permutations

import numpy as np
import pandas as pd

import torch

"""
离线训练
"""
def offline(x,y,test_size=0.3,ap_pairs=10,an_pairs=10):
    data_xy = tuple([x,y])

    # 创建训练和测试集的比例
    train_size = 1 - test_size

    # 训练数据和测试数据集
    triplet_train_pairs = []
    triplet_test_pairs = []

    for data_class in sorted(set(data_xy[1])):

        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where((data_xy[1] != data_class))[0]

        # 生成 anchor 和 positive 对
        A_P_pairs = random.sample(list(permutations(same_class_idx,2)),k=ap_pairs)
        Neg_idx = random.sample(list(diff_class_idx),k=an_pairs)

        # 训练
        A_P_len = len(A_P_pairs)
        Neg_len = len(Neg_idx)

        # anchor 和 positive 对
        for ap in A_P_pairs[:int(A_P_len * train_size)]:
            anchor = data_xy[0][ap[0]]
            positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                negative = data_xy[0][n]
                triplet_train_pairs.append(anchor,positive,negative)

        # 测试
        for ap in A_P_pairs[int(A_P_len*train_size):]:
            anchor = data_xy[0][ap[0]]
            positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                negative = data_xy[0][n]
                triplet_test_pairs.append([anchor,positive,negative])

    return np.array(triplet_train_pairs),np.array(triplet_test_pairs)

"""
计算距离
- 参数: 
- embedding: tensor of shape(batch_size,embed_img)
- squred: Boolean. If true, output
"""



def _get_triplet_mask(labels,device='cpu'):
    """
    选择符合要求的 triplets
    - a,p,n 的 index 
    Args:
        labels:训练数据集的 labels, shape = (batch_size)
    Returns:
        返回 3D mask [a, p, n],对应 triplet (a,p,n) 是 valid 的位置是 True
    """
    
    # 初始化一个二维矩阵，坐标(i,j)不相等设置为 1 
    indices_equal = torch.eye(labels.size()[0]).type(torch.ByteTensor)
    # 得到矩阵是对角线上都是 0 其他位置为 1 的 2 维矩阵
    indices_not_equal = -(indices_equal-1) 

    # 
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k
    
    label_equal = (torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1))
    
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)

    valid_labels = (i_equal_j & ~i_equal_k)

    mask = (distinct_indices.to(device).bool() & valid_labels)

    return mask

# 计算距离两个向量之间距离
def _pairwise_distances(embeddings, squared=False,device='cpu'):

    # 在所有 embedding 间进行点乘

    # shape=(batch_size,features,1)
    embeddings = torch.squeeze(embeddings,1)
    # shape=(batch_size,batch_size)
    dot_product = torch.matmul(embeddings,torch.transpose(embeddings,0,1))
    # 
    square_norm = torch.diag(dot_product)

    distances = torch.unsqueeze(square_norm,1) - 2.0 * dot_product + torch.unsqueeze(square_norm,0)

    distances = torch.max(distances, torch.Tensor([0.0]).to(device))

    # 如何是否有平方
    if not squared:
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)

        distances = distances * (1.0 - mask)

    # 返回
    return distances    
"""
- labels
- embeddings
- margin
- squared
- device: 运行的设备
"""
def online_mine_all(labels, embeddings, margin, squared=False, device='cpu'):

    # 
    pairwise_dist = _pairwise_distances(embeddings, squared=squared, device=device)
    
    # 
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # 损失函数
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # 获取 mask
    mask = _get_triplet_mask(labels, device=device)
    mask = mask.float()
    
    triplet_loss = mask*triplet_loss

    triplet_loss = torch.max(triplet_loss, torch.Tensor([0.0]).to(device))
    valid_triplets = (triplet_loss > 1e-16).float()

    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)

    # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss, num_positive_triplets, num_valid_triplets


def _get_anchor_positive_triplet_mask(labels,device):
    """ 
        Return 2D mask 根据 a 和 p 是唯一的且具有相同的 label 
    
    """
    
    # 检测 i 和 j 是否为唯一
    indices_equal = torch.eye(labels.size()[0])

def _get_anchor_negative_triplet_mask(labels,device):
    pass


def online_mine_hard(labels, embeddings, margin, squared=False, device='cpu'):

    # 获取 pairwise 距离矩阵
    pairwise_dist = _pairwise_distances(embeddings,squared=squared,device=device)

    # 对于每一个 anchor 获取 hardest poisive
    # 对于每一个有效的 positive 收取 mask
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels,device)
    # mask_anchor_positive = mask_anchor_positive


