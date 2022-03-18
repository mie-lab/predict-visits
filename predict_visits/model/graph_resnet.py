from typing import List
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GCNConv, ChebConv


class Kipfblock(torch.nn.Module):
    """GCN Block based on Thomas N Kipf and Max Welling,  Semi-supervised
    classification with graph convolutionalnetworks.arXiv preprint
    arXiv:1609.02907, 2016, https://arxiv.org/abs/1609.02907v4.
    GCN Conv -> Batch Norm  -> RELU -> Dropout
    See http://proceedings.mlr.press/v123/martin20a/martin20a.pdf Figure 3.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 64,
        K: int = 8,
        p: float = 0.5,
        bn: bool = False,
        normalization: str = "sym",
    ):
        """
        Parameters
        ----------
        n_input
            number of input features
        n_hidden: int
            number of output features
        K: int
            Chebyshev filter size :math:`K`. See `torch_geometric.nn.ChebConv`
        p: float
            dropout rate
        bn: bool
            batch normalization?
        """
        super(Kipfblock, self).__init__()
        self.conv1 = ChebConv(
            n_input, n_hidden, K=K, normalization=normalization
        )
        self.p = p
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.do_bn = bn
        if bn:
            self.bn = torch.nn.BatchNorm1d(n_hidden)

    def forward(self, x, edge_index):
        if self.do_bn:
            x = F.relu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.relu(self.conv1(x, edge_index))

        x = F.dropout(x, training=self.training, p=self.p)

        return x


class GraphResnet(torch.nn.Module):
    """Graph resnet based on Martin et al., Graph-ResNets for short-term
    traffic forecasts in almost unknown cities, 2020,
    http://proceedings.mlr.press/v123/martin20a/martin20a.pdf.
    Generalization of http://proceedings.mlr.press/v123/martin20a/martin20a.pdf Figure 3 right.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        nh: Union[int, List[int]] = 42,
        K: int = 4,
        K_mix: int = 2,
        inout_skipconn: bool = False,
        depth: int = 3,
        p: float = 0.5,
        bn: bool = False,
        norm="sym",
        **kwargs
    ):
        """
        Parameters
        ----------
        num_features
        num_classes
        nh: Union[int, List[int]]
            hidden size(s)
        K: int
            Chebyshev filter size :math:`K`. See `torch_geometric.nn.ChebConv`
        K_mix:
            Chebyshev filter size for `inout_skipconn`.
        inout_skipconn: bool
        depth: int
        p: float
            dropout rate
        bn: bool
            batch normalization?
        """
        super(GraphResnet, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.depth = depth

        self.Kipfblock_list = nn.ModuleList()
        self.skipproject_list = nn.ModuleList()

        if isinstance(nh, list):
            # if you give every layer a different number of channels
            # you need one number of channels for every layer!
            assert len(nh) == depth

        else:
            channels = nh
            nh = []
            for _ in range(depth):
                nh.append(channels)

        for i in range(depth):
            if i == 0:
                self.Kipfblock_list.append(
                    Kipfblock(
                        n_input=num_features,
                        n_hidden=nh[0],
                        K=K,
                        p=p,
                        bn=bn,
                        normalization=norm,
                    )
                )
                self.skipproject_list.append(
                    ChebConv(num_features, nh[0], K=1, normalization=norm)
                )
            else:
                self.Kipfblock_list.append(
                    Kipfblock(
                        n_input=nh[i - 1],
                        n_hidden=nh[i],
                        K=K,
                        p=p,
                        bn=bn,
                        normalization=norm,
                    )
                )
                self.skipproject_list.append(
                    ChebConv(nh[i - 1], nh[i], K=1, normalization=norm)
                )

        if inout_skipconn:
            self.conv_mix = ChebConv(
                nh[-1] + num_features, num_classes, K=K_mix, normalization=norm
            )
        else:
            self.conv_mix = ChebConv(
                nh[-1], num_classes, K=K_mix, normalization=norm
            )

    def forward(self, data, **kwargs):

        x, edge_index = data.x, data.edge_index

        for i in range(self.depth):
            x = self.Kipfblock_list[i](x, edge_index) + self.skipproject_list[
                i
            ](x, edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)

        return x


class VisitPredictionModel(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        out_dim=1,
        graph_enc_dim=64,
        graph_layers: Union[int, List[int]] = 42,
        graph_k: int = 4,
        inp_embed_dim=64,
        ff_layers: List[int] = [64, 32],
        adj_is_symmetric=True,
        dropout_prob=0,
        **kwargs
    ):
        """Adjecency dim (= number of nodes) only required for autoencoder"""
        super(VisitPredictionModel, self).__init__()
        self.norm = "sym" if adj_is_symmetric else "rw"
        self.dropout_prob = dropout_prob

        # graph processing
        self.graph_module = GraphResnet(
            node_feat_dim,
            graph_enc_dim,
            K=graph_k,
            nh=graph_layers,
            norm=self.norm,
            p=dropout_prob,
            **kwargs,
        )
        # second input processing:
        self.embed_layer = nn.Linear(node_feat_dim - 1, inp_embed_dim)
        # first forward layer
        first_layer_size = graph_enc_dim + inp_embed_dim
        self.ff_layers = nn.ModuleList(
            [nn.Linear(first_layer_size, ff_layers[0])]
        )
        for i in range(len(ff_layers) - 1):
            self.ff_layers.append(nn.Linear(ff_layers[i], ff_layers[i + 1]))
        # last layer
        self.ff_layers.append(nn.Linear(ff_layers[-1], out_dim))

    def forward(self, data):
        # 1. Graph processing - node embeddings
        graph_embed = self.graph_module(data)
        graph_embed = global_mean_pool(graph_embed, data.batch)

        # 2. processing of separate input
        separate_input = data.y[:, :-1]
        inp_embed = self.embed_layer(separate_input)

        # concat both
        x = torch.cat((graph_embed, inp_embed), dim=1)

        # 3. feed forward:
        for layer in self.ff_layers:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
            x = layer(x)

        out = torch.sigmoid(x)
        return out


if __name__ == "__main__":
    from dataset import MobilityGraphDataset
    import torch_geometric

    dataset = MobilityGraphDataset(["t120_gc2_poi.pkl"])

    model = VisitPredictionModel(25)

    loader = torch_geometric.loader.DataLoader(dataset, batch_size=2)
    counter = 0
    for test in loader:
        print(test.x.size())
        print(test.y.size())
        print(test.edge_index.size())
        print(test.edge_attr.size())
        # print(test.batch)
        out = model(test)
        print("OUT", out.size())
        print()
        counter += 1
        if counter > 3:
            break
