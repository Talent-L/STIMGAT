import torch
import torch.nn as nn
import torch.nn.functional as F

def Nor(tensor: torch.Tensor):
    """
    将数据放到[-1, 1]之间
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    scaled_tensor = 2 * (tensor - tensor_min) / (tensor_max - tensor_min) - 1
    return scaled_tensor


def Loss_bal(fea: torch.Tensor, dis_mat: torch.Tensor, out_mat:torch.Tensor, method: str='enhance'):
    """
    平衡损失
    fea: 原始特征矩阵
    out_mat: GAT还原矩阵
    """
    fea_dis = torch.cdist(fea, fea, p=2)
    fea_dis_mark = Nor(fea_dis)
    # is_mat_mark = dis_mat - 0.01
    dis_mat_mark = Nor(dis_mat)
    mark = torch.mul(fea_dis_mark, dis_mat_mark)
    if method == 'enhance':
        mark = torch.clamp(mark, max=0)
    if method == 'weaken':
        mark = torch.clamp(mark, min=0)
    mark = torch.abs(mark)
    
    if method == 'enhance' or method == 'weaken':
        # row = torch.ceil(torch.sum(mark, dim=0)).to(torch.int64)
        row = torch.sum(mark, dim=0).to(torch.int64)
        #col = torch.ceil(torch.sum(mark, dim=1)).to(torch.int64)
        row = row / row.max()
        #col = col / col.max() 

        loss_bal = F.mse_loss(fea*row.unsqueeze(1), out_mat*row.unsqueeze(1))
        #print(F.mse_loss(fea*row.unsqueeze(1), out_mat*row.unsqueeze(1))==F.mse_loss(fea*col.unsqueeze(1), out_mat*col.unsqueeze(1)))
        return loss_bal