import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import random
import scipy.sparse as sp

import torch
from torch import nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True



from STIMGAT import STIMGAT
from CL_module import permutation, add_contrastive_label, construct_interaction
from utils import Transfer_pytorch_Data
from Loss import Loss_bal



def train_STIMGAT(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STIMGAT', method='weaken',
                gradient_clipping=5.,  weight_decay=0.0001, theta=0.8, alpha=0.1, Con=False, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                image_data=torch.empty(0)):
    """
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    theta
        balance parameter for the loss function.
    save_loss
        If True, the training loss is saved in adata.uns['STIMGAT_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STIMGAT_ReX'].
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed=random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 稀疏矩阵储存为csr格式，提高效率
    adata.X = sp.csr_matrix(adata.X)
    
    # 如果表达矩阵中含有高变基因
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    # 维度输出以及处理问题
    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    # 将anndata数据转为torch
    data = Transfer_pytorch_Data(adata_Vars)
    adata.uns['Dis_mat'] = adata_Vars.uns['Dis_mat']
    
    # +操作符做的是连接，默认参数隐藏层数量为512，输出维度为30
    model = STIMGAT(hidden_dims = [data.x.shape[1]] + hidden_dims).to(device)
    data = data.to(device)

    # 对比学习模块
    if Con==True:
        loss_CSL = nn.BCEWithLogitsLoss()
        feat_a = permutation(data.x).to(device)
        
        label_CSL = add_contrastive_label(data.x.shape[0])#.to(device)
        label_CSL = torch.FloatTensor(label_CSL).to(device)
        
        adj = construct_interaction(adata)
        adj = torch.FloatTensor(adj).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #loss_list = []
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        
        if Con==True:
            # print(data.x.device, data.edge_index.device, feat_a.device, adj.device)
            z, out, ret, ret_a = model(data.x, data.edge_index, Con=True, Con_data_a=feat_a, adj=adj)
        else:
            z, out = model(data.x, data.edge_index)
        
        loss_bal = Loss_bal(data.x, data.dis_mat, out, method=method)    
        loss = F.mse_loss(data.x, out) + theta*loss_bal #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
        if Con==True:
            loss_sl_1 = loss_CSL(ret, label_CSL)
            loss_sl_2 = loss_CSL(ret_a, label_CSL)
            loss = loss + alpha*(loss_sl_1 + loss_sl_2)

        #loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    
    if image_data.numel() > 0:
        model.eval()
        z, out = model(data.x, data.edge_index)
        image_z, image_out = model(image_data, data.edge_index)
        z = 0.7*z + 0.3*image_z
        out = out + image_out

        STIMGAT_rep = z.to('cpu').detach().numpy()
        adata.obsm[key_added] = STIMGAT_rep
    
    else:
        model.eval()
        z, out = model(data.x, data.edge_index)
    
        STIMGAT_rep = z.to('cpu').detach().numpy()
        adata.obsm[key_added] = STIMGAT_rep

    if save_loss:
        adata.uns['STIMGAT_loss'] = loss
    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX<0] = 0
        adata.layers['STIMGAT_ReX'] = ReX

    return adata