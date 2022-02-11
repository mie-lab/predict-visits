import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    # TODO: Could be replaced by GCNConv by torch geometric module
    """

    def __init__(self, in_features, out_features, dropout=0.0, act=torch.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inp, adj):
        inp = F.dropout(inp, self.dropout, self.training)
        support = torch.matmul(inp, self.weight)
        output = torch.matmul(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class ClassificationModel(nn.Module):
    def __init__(
        self,
        graph_feat_dim,
        loc_feat_dim,
        adj_dim=None,
        hidden_gc_1=32,
        hidden_gc_2=64,
        hidden_dec_1=64,
        hidden_dec_2=32,
        dropout=0,
    ):
        """Adjecency dim (= number of nodes) only required for autoencoder"""
        super(ClassificationModel, self).__init__()
        self.adj_dim = adj_dim
        self.gc1 = GraphConvolution(
            graph_feat_dim, hidden_gc_1, dropout, act=torch.relu
        )
        self.gc2 = GraphConvolution(
            hidden_gc_1, hidden_gc_2, dropout, act=lambda x: x
        )
        # self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # decoding is simply inner product?!
        self.dec_hidden_1 = nn.Linear(hidden_gc_2 + loc_feat_dim, hidden_dec_1)
        self.dec_hidden_2 = nn.Linear(hidden_dec_1, hidden_dec_2)
        self.dec_out = nn.Linear(hidden_dec_2, 1)

    def graph_processing(self, x, adj):
        """Pass graph through two conv layers and sum up node-wise features"""
        hidden1 = self.gc1(x, adj)
        z = self.gc2(hidden1, adj)
        z_pooled = torch.sum(z, dim=1)
        return z_pooled

    def feed_forward(self, z):
        """Pass graph embedding and new feature vector through MLP"""
        hidden_dec_1 = torch.relu(self.dec_hidden_1(z))
        hidden_dec_2 = torch.relu(self.dec_hidden_2(hidden_dec_1))
        out_dec = self.dec_out(hidden_dec_2)
        return out_dec

    def forward(self, x, adj, input_2):
        # feed graph through conv layers
        graph_output = self.graph_processing(x, adj)
        # concat with new node
        together = torch.cat((graph_output, input_2), dim=1)
        # pass through feed forward network
        out = torch.sigmoid(self.feed_forward(together))
        return out


if __name__ == "__main__":
    model = ClassificationModel(10, 9)
    import random
    import numpy as np

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    inp1 = torch.randn(1, 50, 10).repeat(3, 1, 1)
    print(inp1.size())
    inp2 = torch.randn(1, 50, 50).repeat(3, 1, 1)
    inp3 = torch.randn(1, 9).repeat(3, 1)

    out = model(inp1, inp2, inp3)

    # from torch_geometric.nn.conv import GCNConv

    # conv_test = GCNConv(10, 20)
    # row, col, edge_attr = inp2.t().coo()
    # edge_index = torch.stack([row, col], dim=0)

    # out = conv_test(edge_index, edge_attr)
    print(out)
