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
        support = torch.mm(inp, self.weight)
        output = torch.spmm(adj, support)
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


class ClassificationModelV1(nn.Module):
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
        relative_feats=False,
    ):
        """Adjecency dim (= number of nodes) only required for autoencoder"""
        super(ClassificationModelV1, self).__init__()
        self.adj_dim = adj_dim
        self.relative_feats = relative_feats
        self.gc1 = GraphConvolution(
            graph_feat_dim, hidden_gc_1, dropout, act=torch.relu
        )
        self.gc2 = GraphConvolution(
            hidden_gc_1, hidden_gc_2, dropout, act=lambda x: x
        )
        self.dec_hidden_1 = nn.Linear(hidden_gc_2 + loc_feat_dim, hidden_dec_1)
        self.dec_hidden_2 = nn.Linear(hidden_dec_1, hidden_dec_2)
        self.dec_out = nn.Linear(hidden_dec_2, 1)

    def graph_processing(self, x, adj):
        """Pass graph through two conv layers and sum up node-wise features"""
        hidden1 = self.gc1(x, adj)
        z = self.gc2(hidden1, adj)
        z_pooled = torch.sum(z, dim=0)
        return z_pooled

    def feed_forward(self, z):
        """Pass graph embedding and new feature vector through MLP"""
        hidden_dec_1 = torch.relu(self.dec_hidden_1(z))
        hidden_dec_2 = torch.relu(self.dec_hidden_2(hidden_dec_1))
        out_dec = self.dec_out(hidden_dec_2)
        return out_dec

    def forward(self, node_feats, adj, new_loc_feats):
        if self.relative_feats:
            node_feats[:, :-1] = node_feats[:, :-1] - torch.unsqueeze(
                new_loc_feats, 0
            )
            # TODO: add relative one as separate input stream?
        # feed graph through conv layers
        graph_output = self.graph_processing(node_feats, adj)
        # concat with new node
        together = torch.cat((graph_output, new_loc_feats), dim=0)
        # pass through feed forward network
        out = torch.sigmoid(self.feed_forward(together))
        return out
