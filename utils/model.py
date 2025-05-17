import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import GCN2Conv
from .sublayers import Attention,MLP_encoder,MLP_classifier
import math


class LLM_RC(nn.Module):
    # def __init__(self, data, num_classes, hid: int = 128, dropout=0.5, use_bn = True):
    def __init__(self, data, num_classes, hid: int = 128, dropout=0.5, use_bn=True, use_evl=True):
        super(LLM_RC, self).__init__()
        torch.manual_seed(9999)
        torch.cuda.manual_seed(9999)
        
        # 动态获取每个视图的特征维度
        self.view_dims = {}
        for view_id, dim in data.view_features.items():
            self.view_dims[view_id] = dim
        
        self.hid = hid
        self.num_classes = num_classes
        
        # 为每个视图动态创建编码器
        self.encoders = nn.ModuleDict()
        for view_id, dim in self.view_dims.items():
            self.encoders[f"encoder_{view_id}"] = MLP_encoder(
                nfeat=dim,
                nhid=self.hid,
                ncla=self.num_classes,
                dropout=dropout,
                use_bn=use_bn
            )
        
        self.classifier = MLP_classifier(
            nfeat=self.hid,
            nclass=self.num_classes,
            dropout=dropout
        )
        self.use_evl = use_evl
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

    def Simple(self, e1, e2,e3):
        return (e1+e2+e3)/3

    def forward(self, x_list):
        # x_list是一个视图特征的列表
        evidences = []
        outputs = []
        alphas = []
        for i, view_id in enumerate(sorted(self.view_dims.keys())):
            # evidence = F.softplus(self.encoders[f"encoder_{view_id}"](x_list[i]))
            # evidences.append(evidence)
            # alphas.append(evidence + 1)

            out = self.encoders[f"encoder_{view_id}"](x_list[i])
            if self.use_evl:
                out = F.softplus(out) + 1  # Dirichlet alpha
            else:
                out = F.softplus(out)
            outputs.append(out)
        alphas = outputs
        if len(alphas) == 1:
            return tuple(alphas+alphas)
        
        if self.use_evl:
            combined_alpha = self.DS_Combin_two(alphas[0], alphas[1])
            for i in range(2, len(alphas)):
                combined_alpha = self.DS_Combin_two(combined_alpha, alphas[i])
        else:
            combined_alpha = self.Simple(outputs[0], outputs[1], outputs[2])

        return tuple(outputs + [combined_alpha])

        # return tuple(alphas + [combined_alpha])

    # def forward(self, x_list):
    #     # x_list 是一个视图特征的列表
    #     outputs = []

    #     # 每个视图直接输出 logits（不使用 softplus 和 evidence）
    #     for i, view_id in enumerate(sorted(self.view_dims.keys())):
    #         logits = F.softplus(self.encoders[f"encoder_{view_id}"](x_list[i]))  # shape: [batch_size, num_classes]
    #         outputs.append(logits)

    #     # 如果只有一个视图，直接返回重复的输出
    #     if len(outputs) == 1:
    #         return tuple(outputs + outputs)

    #     # 否则执行简单平均融合
    #     combined_output = self.Simple(outputs[0], outputs[1], outputs[2])

    #     # 返回所有单视图输出 + 融合结果
    #     return tuple(outputs + [combined_output])

class RCML(nn.Module):
    def __init__(self, num_views, dims, num_classes): # self, data, num_classes, hid: int = 128, dropout=0.5, use_bn = True
        super(RCML, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])

    def forward(self, X):
        # get evidence
        evidences = dict()
        for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])
        # average belief fusion
        evidence_a = evidences[0]
        for i in range(1, self.num_views):
            evidence_a = (evidences[i] + evidence_a) / 2
        return evidences, evidence_a


class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h


