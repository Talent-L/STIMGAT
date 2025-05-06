import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from gat_conv import GATConv
from CL_module import AvgReadout, Discriminator

class STIMGAT(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STIMGAT, self).__init__()

        # 输入维度、隐藏层维度、输出维度
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.out_dim = out_dim

        # head——注意力头数量
        # concat=False, 不将多头输出特征拼接，而是求和
        # dropout=0,无 Dropout
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        # 聚合节点方式
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

    def forward(self, features, edge_index, Con=False, Con_data_a=False, adj=False, 
                device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        # feature特征矩阵, edge_index图的边索引
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        if Con==True:

            h1_a = F.elu(self.conv1(features, edge_index))
            # h1_a = F.elu(self.conv1(Con_data_a, edge_index))
            h2_a = self.conv2(h1_a, edge_index, attention=False)

            g = self.read(h2, adj)
            g = self.sigm(g) 
            
            g_a = self.read(h2_a, adj)
            g_a = self.sigm(g_a) 

            disc = Discriminator(self.out_dim).to(device) 
            ret = disc(g, h2, h2_a) 
            ret_a = disc(g_a, h2_a, h2) 

            return h2, h4, ret, ret_a 

        return h2, h4  # F.log_softmax(x, dim=-1)
    
    def image_gene(self, features, edge_index):
        """
        还原图像信息
        """
        h3 = F.elu(self.conv3(features, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)
        
        return h4