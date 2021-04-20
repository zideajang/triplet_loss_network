import torch

"""


"""

def triplet_loss(anchor,positive,negative,alpha=0.4,device='cuda:0'):
    """
    实现 Triplet Loss 损失函数
    y_true: 样本标签
    embedding: 向量
    - anchor: 
    - positive:
    - negative:
    """

    # 计算 anchor 到 positive 之间的距离
    pos_dist = torch.sum((anchor - positive).pow(2),axis=1)

    # 计算 anchor 到 negative 之间的距离
    neg_dist = torch.sum((anchor-negative).pow(2),axis=1)

    # 计算 loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = torch.max(basic_loss,torch.tensor([0],device=device).float())

    return loss


