import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class Linear(nn.Module): 
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class MLP_classifier(nn.Module):#
    def __init__(self, nfeat, nclass, dropout):
        super(MLP_classifier, self).__init__()
        self.Linear1 = Linear(nfeat, nclass, dropout, bias=True)

    def forward(self, x):
        out = self.Linear1(x)
        return torch.log_softmax(out, dim=1), out

    
class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, ncla, dropout, use_bn):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid*2, dropout, bias=True)
        self.Linear2 = Linear(nhid*2, nhid, dropout, bias=True)
        self.Linear3 = Linear(nhid, ncla, dropout, bias=True)
        # self.Linear = Linear(nfeat, ncla, dropout, bias=True)

        self.use_bn = use_bn 
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nfeat)
            self.bn2 = nn.BatchNorm1d(nhid*2)
            self.bn3 = nn.BatchNorm1d(nhid)

    def forward(self, x):
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(self.Linear1(x))
        # if self.use_bn:
        #     x = self.bn2(x)
        x = F.relu(self.Linear2(x))
        if self.use_bn:
            x = self.bn3(x)
        x = F.relu(self.Linear3(x))
        # x = self.Linear(x)

        return x

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        attention_weights = self.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.size(-1)))

        out = attention_weights @ v
        return out


class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.lin1(x).relu()
        h = self.lin2(h).relu()

        output = self.out_layer(h).squeeze()
        return output
    

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
       
        out = self.classifier(h).squeeze()
        return out


class GCN_EW(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index):
        super().__init__()
        torch.manual_seed(1234)
        self.edge_weight = torch.nn.Parameter(torch.zeros(edge_index.shape[1]))

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index, torch.exp(self.edge_weight)).relu()
        h = self.conv2(h, edge_index, torch.exp(self.edge_weight)).relu()

        out = self.classifier(h).squeeze()
        return out


class GAT(nn.Module):
    def __init__(self, hidden_channels, heads, in_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GATConv(in_dim, hidden_channels, heads)
        self.conv2 = GATConv(heads*hidden_channels, hidden_channels, heads)
        self.classifier = nn.Linear(heads*hidden_channels, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()

        out = self.classifier(h).squeeze()
        return out
    