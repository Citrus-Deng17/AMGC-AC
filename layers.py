import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    基础图卷积层 (GCN Layer)
    适用于对构建好的第一组学特征邻接矩阵 (fadj) 和特征矩阵 (X) 进行特征提取。
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input: 节点特征矩阵 [N, in_features]
        # adj: 特征图邻接矩阵 [N, N]
        support = torch.mm(input, self.weight)

        if adj.is_sparse:
            output = torch.spmm(adj, support)
        else:
            output = torch.mm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class SGC(Module):
    """
    简化图卷积层 (Simple Graph Convolution Layer)
    """

    def __init__(self, in_features, out_features, K=1, bias=True):
        super(SGC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = input
        # 进行 K 次图信号平滑聚合
        for _ in range(self.K):
            if adj.is_sparse:
                output = torch.spmm(adj, output)
            else:
                output = torch.mm(adj, output)

        output = torch.mm(output, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'