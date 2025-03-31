import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import GCN2Conv
from .sublayers import Attention,MLP_encoder,MLP_classifier
import math


class LLM_RC(nn.Module):
    def __init__(self, data, num_classes, v=0, att=False, sim=False, pse=True, hid: int = 128, dropout=0.5, use_bn = True):
        super(LLM_RC, self).__init__()
        torch.manual_seed(9999)
        torch.cuda.manual_seed(9999)
        self.feat_1 = data.view1_features
        self.feat_2 = data.view2_features
        self.hid = hid
        self.num_classes = num_classes
        self.att = att
        self.sim = sim
        self.v = 0
        self.encoder = MLP_encoder(nfeat=self.feat_1,
                                 nhid=self.hid,
                                 ncla=self.num_classes,
                                 dropout=dropout,
                                 use_bn=use_bn)
        self.encoder_2 = MLP_encoder(nfeat=self.feat_2,
                                 nhid=self.hid,
                                 ncla=self.num_classes,
                                 dropout=dropout,
                                 use_bn=use_bn)

        # self.encoder = MLP_encoder(nfeat=self.feat,
        #                     nhid=self.hid,
        #                     dropout=dropout)
        # self.encoder_pse = MLP_encoder(nfeat=self.feat*2,
        #             nhid=self.hid*2,
        #             dropout=dropout)

        # self.classifier_pse = MLP_classifier(nfeat=self.hid*2,
        #             nclass=self.num_classes,
        #             dropout=dropout)

        self.classifier = MLP_classifier(nfeat=self.hid,
                                    nclass=self.num_classes,
                                    dropout=dropout)

        self.use_bn = use_bn
        self.attention = Attention(input_dim=self.num_classes)

    def DS_Combin_two(self, alpha1, alpha2):
        """ 
        :(batch_size, num_classes)，表示批次中的每个样本在每个类别下的 Dirichlet 分布参数。
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        classes = self.num_classes
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            # print(type(alpha[v]), alpha[v])
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = classes/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def Simple(self, e1, e2):
        return (e1+e2)/2

    def forward(self, x, x_glo):

        evidence = F.softplus(self.encoder(x))
        evidence_glo = F.softplus(self.encoder_2(x_glo))


        if self.v == '0':
            alpha_1,alpha_2 = evidence+1, evidence_glo+1
            alpha_all = self.DS_Combin_two(alpha_1,alpha_2)
            return alpha_1,alpha_2,alpha_all
        

        if self.att:
            att_fuse = self.attention(evidence, evidence_glo)
            return att_fuse
        elif self.sim:
            sim_fuse = self.Simple(evidence, evidence_glo)
            return sim_fuse
        
        alpha_1,alpha_2= evidence+1, evidence_glo+1
        alpha_all = self.DS_Combin_two(alpha_1,alpha_2)

        return alpha_1,alpha_2,alpha_all



