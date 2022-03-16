from turtle import pos
from scipy.sparse import coo
import torch
import pickle
import os
import scipy.sparse as sp
import numpy as np
from torch.functional import _return_counts, norm
import torch_geometric
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix

from predict_visits.utils import *
from predict_visits.geo_embedding.embed import (
    std_log_embedding,
    sinusoidal_embedding,
)
from predict_visits.features.purpose import purpose_df_to_matrix


class MobilityGraphDataset(InMemoryDataset):
    """
    Dataset to yield graph and test nodes separately
    -> create new dataset if we want to do link prediction / other pipeline

    Arguments:
        dataset_files: list of str
            List of pkl files that are combined in this dataset, e.g.["gc1.pkl"]
        path: str
            Path where the dataset files are stored
        ratio_predict: float (between 0 and 1, default 0.1)
            Ratio of nodes per graph that is left out for training
        label_cutoff: float (between 0 and 1, default 0.95)
            Only the nodes in the lower x-quantile are left out for
            training (e.g. home node can not be left out)
        nr_keep: graphs are reduced to the x nodes with highest degree
    """

    def __init__(
        self,
        dataset_files,
        device=torch.device("cpu"),
        root="data",
        transform=None,
        pre_transform=None,
        ratio_predict=0.1,
        label_cutoff=10,
        nr_keep=50,
        min_label=1,
        log_labels=False,
        adj_is_unweighted=True,
        adj_is_symmetric=True,
        relative_feats=False,
        embedding="simple",
        **kwargs,
    ):
        """
        Data Loader for mobility graphs
        """
        super(MobilityGraphDataset, self).__init__(
            root, transform, pre_transform
        )
        self.relative_feats = relative_feats
        self.adj_is_unweighted = adj_is_unweighted
        self.adj_is_symmetric = adj_is_symmetric
        self.nr_keep = nr_keep
        self.ratio_predict = ratio_predict
        self.min_label = min_label
        self.device = device
        # Load data - Note: adjacency is a list of adjacency matrices, and
        # coordinates is a list of arrays (one for each user)
        (self.users, adjacency_graphs, node_feat_list) = self.load(
            dataset_files, root
        )

        # Preprocess the graph to get adjacency and node features
        self.node_feats = []
        self.adjacency = []
        for i in range(len(self.users)):
            # process all graphs
            nf_one, adj_one, _ = self.graph_preprocessing(
                adjacency_graphs[i],
                node_feat_list[i],
                label_cutoff=label_cutoff,
                nr_keep=nr_keep,
                log_labels=log_labels,
                embedding=embedding,
            )
            self.node_feats.append(nf_one)
            self.adjacency.append(adj_one)

        self.nr_graphs = len(self.adjacency)

        # store the feature dimension and normalization stats
        self.num_feats = self.node_feats[0].shape[1]
        print("Number samples after preprocessing", len(self.node_feats))

    def split_graphs_v1(self, node_feats, adjacency):
        """
        Version 1
        Don't select left-out nodes at runtime but rather before
        """
        # Split into graph (adjacency and known node feats) and "new" locations
        # (predict_node_feats) with their visit numbers (labels)
        self.adjacency, self.known_node_feats = [], []
        self.predict_node_feats, self.labels = [], []
        for i in range(len(node_feats)):
            feat_vector = node_feats[i]

            # Leave out some nodes from the graph --> only take the ones that
            # are visited less frequently --> labs preprocessed such that cutoff
            # is at 1
            label_vector = feat_vector[:, -1]
            possible_nodes = np.where((label_vector <= 1) & (label_vector > 0))[
                0
            ]
            rand_perm = np.random.permutation(possible_nodes)
            # we use ratio_predict percent of the nodes to predict
            number_nodes_predict = int(len(feat_vector) * self.ratio_predict)
            # divide in known nodes and the ones to predict:
            predict_nodes = rand_perm[:number_nodes_predict]
            # known nodes are also the rest (that is not in possible_nodes)
            known_nodes = list(rand_perm[number_nodes_predict:]) + list(
                np.where(feat_vector[:, -1] > 1)[0]
            )

            # reduce node feats to known nodes
            self.known_node_feats.append(feat_vector[known_nodes])
            self.predict_node_feats.append(feat_vector[predict_nodes, :-1])
            self.labels.append(feat_vector[predict_nodes, -1])

            # reduce adjacency matrix to known nodes and preprocess
            adj = adjacency[i][known_nodes]
            adj = adj[:, known_nodes]
            self.adjacency.append(self.adjacency_preprocessing(adj).float())

        self.avail_predict_nodes = [len(arr) for arr in self.labels]

    def load(self, dataset_files, path):
        users, adjacency_graphs, node_feat_list = [], [], []
        for data_file in dataset_files:
            with open(os.path.join(path, data_file), "rb") as infile:
                (user_this, adjacency_this, node_feats_this) = pickle.load(
                    infile
                )
            users.extend(user_this)
            adjacency_graphs.extend(adjacency_this)
            node_feat_list.extend(node_feats_this)
            print(f"Loaded {len(user_this)} graphs from dataset {data_file}")
        print("Overall dataset size:", len(users))
        return users, adjacency_graphs, node_feat_list

    @staticmethod
    def adjacency_preprocessing(adjacency_matrix):
        """
        Preprocess the adjacency matrix - return unweighted normalized tensor
        """
        # GCN requires an symmetric adjacancy matrix. We could check for models
        # designed for directed graphs later on
        symmetric_adjacency = adjacency_matrix + adjacency_matrix.T
        unweighted_adjacency = (symmetric_adjacency > 0).astype(int)
        return sparse_mx_to_torch_sparse_tensor(
            preprocess_adj(unweighted_adjacency)
        )

    @staticmethod
    def node_feature_preprocessing(
        node_feat_df,
        stats=None,
        embedding="simple",
        include_purpose=True,
        include_dist=True,
        include_poi=True,
        include_time=False,
    ):
        # transform geographic_coordinates
        coords_raw = np.array(node_feat_df[["x_normed", "y_normed"]])
        if embedding == "simple":
            embedded_coords, stats = std_log_embedding(coords_raw, stats)
        elif embedding == "none":
            embedded_coords = coords_raw
        elif embedding == "sinus":
            embedded_coords = sinusoidal_embedding(coords_raw, 10)
        else:
            raise ValueError(
                f"Wrong embedding type {embedding}, must be simple or sinus"
            )
        # collect features in a list:
        features_to_include = [embedded_coords]

        # purpose feature
        if include_purpose:
            purpose_feature_arr = purpose_df_to_matrix(node_feat_df)
            features_to_include.append(purpose_feature_arr)

        # distance feature (although already contained in coordinates)
        if include_dist:
            distance_arr = np.expand_dims(
                np.log10(node_feat_df["distance"].values / 1000 + 1), 1
            )
            features_to_include.append(distance_arr)

        # Average start time (average is a very bad feature, refine that later)
        # use sinuoidal transform to express 0:00 == 23:59:99
        if include_time:
            started = node_feat_df["started_at"].values
            sin_start = np.sin(2 * np.pi * started / 24)
            cos_start = np.cos(2 * np.pi * started / 24)
            start_time_arr = np.vstack([sin_start, cos_start]).swapaxes(1, 0)
            features_to_include.append(start_time_arr)

        # POI data
        if include_poi:
            poi_values = np.array(list(node_feat_df["poiRep"].values))
            assert len(poi_values.shape) == 2, "Problem: NaNs in poi values"
            features_to_include.append(poi_values)

        feature_matrix = np.hstack(features_to_include)
        return feature_matrix, stats

    @staticmethod
    def graph_preprocessing(
        adjacency,
        node_feature_df,
        label_cutoff=10,
        nr_keep=50,
        dist_thresh=500,
        log_labels=False,
        embedding="simple",
        **kwargs,
    ):
        """
        Preprocess the node features of the graph
        dist_thresh: Maximum distance of location from home (in km)
        """
        # 1) process node features
        # transform geographic coordinates and make feature matrix
        (
            feature_matrix,
            feat_stats,
        ) = MobilityGraphDataset.node_feature_preprocessing(
            node_feature_df, embedding=embedding
        )
        # NOTE: out-degree is in-degree!
        label = node_feature_df["out_degree"].values

        # 2) crop or pad adjacency matrix to the x nodes with highest degree
        overall_degree = (
            np.array(np.sum(adjacency, axis=0))[0]
            + np.array(np.sum(adjacency, axis=1))[0]
        )
        # additionally, filter out locs w distance higher than dist_thresh
        too_far_away = np.where(
            node_feature_df["distance"].values > dist_thresh * 1000
        )[0]
        sorted_usable_nodes = [
            ind for ind in np.argsort(overall_degree) if ind not in too_far_away
        ]
        use_nodes = sorted_usable_nodes[-nr_keep:]
        # 2.1) crop adjacency
        adj_crop = adjacency[use_nodes]
        adj_crop = adj_crop[:, use_nodes]

        # 2.2) restrict features to use_nodes
        feature_matrix = feature_matrix[use_nodes]

        # 3) Get label (number of visits)
        # restrict labels
        label = label[use_nodes]
        # upper bound on labels can either be <1 --> quantile or the upper
        # boudn directly
        cutoff = MobilityGraphDataset.prep_cutoff(
            label, label_cutoff, log_labels
        )
        # normalize labels
        label = MobilityGraphDataset.norm_label(
            label, cutoff, log_labels=log_labels
        )

        # concatenate the other features with the labels and append
        concat_feats_labels = np.concatenate(
            (feature_matrix, np.expand_dims(label, 1)), axis=1
        )

        return concat_feats_labels, adj_crop, [feat_stats, cutoff]

    @staticmethod
    def prep_cutoff(label, label_cutoff, log_labels=False):
        if label_cutoff < 1:
            cutoff = max([np.quantile(label, label_cutoff), 1])
        else:
            cutoff = label_cutoff
        if log_labels:
            cutoff = np.log(cutoff + 1)
        return cutoff

    @staticmethod
    def norm_label(label, label_cutoff, log_labels=False):
        if log_labels:
            label = np.log(label + 1)
        # normalize label by the quantile --> value between 0 and 1
        label = label / label_cutoff
        return label

    @staticmethod
    def unnorm_label(normed_label, label_cutoff, log_labels=False):
        label = normed_label * label_cutoff
        if log_labels:
            label = np.exp(label) - 1
        return label

    def len(self):
        return self.nr_graphs

    @staticmethod
    def transform_to_torch(
        adj,
        known_node_feats,
        predict_node_feats,
        relative_feats=False,
        adj_is_unweighted=True,
        adj_is_symmetric=True,
        add_batch=False,
    ):
        if adj_is_symmetric:
            adj = adj + adj.T
        if adj_is_unweighted:
            adj = (adj > 0).astype(int)

        # transform to index & attr
        edge_index, edge_attr = from_scipy_sparse_matrix(adj)

        # transform to torch
        known_node_feats = torch.from_numpy(known_node_feats)
        predict_node_feats = torch.from_numpy(predict_node_feats)

        # get features of new (test) location
        label_node_feats = torch.unsqueeze(predict_node_feats.float(), 0)

        # transform node features to relative node features wrt new loc
        if relative_feats:
            known_node_feats[:, :-1] = (
                known_node_feats[:, :-1] - label_node_feats[:, :-1]
            )

        data_sample = torch_geometric.data.Data(
            x=known_node_feats.float(),
            edge_index=edge_index,
            edge_attr=edge_attr.float(),
            y=label_node_feats,
        )
        if add_batch:
            data_sample.batch = torch.tensor(
                [0 for _ in range(known_node_feats.size()[0])]
            )
        return data_sample

    @staticmethod
    def select_test_node(node_feat, adj, min_lab=0):
        label_col = node_feat[:, -1]
        # Divide into the known and unkown nodes
        possible_nodes = np.where((label_col >= min_lab) & (label_col <= 1))[0]
        if len(possible_nodes) == 0:
            # if doesn't work, just pick any (should not happen often)
            predict_node = np.random.randint(0, len(node_feat))
        else:
            predict_node = np.random.choice(possible_nodes)

        known_nodes = [i for i in range(len(node_feat)) if i != predict_node]

        # restrict feats to known nodes and convert to torch
        known_node_feats = node_feat[known_nodes]
        # new node features and label
        predict_node_feats = node_feat[predict_node, :]
        # reduce adjacency matrix to known nodes and preprocess
        adj = adj[known_nodes]
        adj = adj[:, known_nodes]

        return adj, known_node_feats, predict_node_feats, predict_node

    def get(self, idx):
        adj = self.adjacency[idx]
        node_feat = self.node_feats[idx]

        # select one node as test node and remove it from the graph
        (adj, known_node_feats, predict_node_feats, _) = self.select_test_node(
            node_feat, adj
        )

        # transform to pytorch geometric data
        data_sample = self.transform_to_torch(
            adj, known_node_feats, predict_node_feats, self.relative_feats
        ).to(self.device)

        return data_sample

    # Version 1: get one of the left out nodes
    # def __len__(self):
    #     return self.nr_graphs * self.test_nodes_per_graph

    # def __getitem__(self, idx_in):
    #     idx = idx_in // self.test_nodes_per_graph
    #     lab_idx = idx_in % self.test_nodes_per_graph
    #     return (
    #         self.adjacency[idx],
    #         self.known_node_feats[idx],
    #         self.predict_node_feats[idx][lab_idx],
    #         self.labels[idx][lab_idx],
    #     )


if __name__ == "__main__":
    dataset = MobilityGraphDataset(["t120_gc2_poi.pkl"])
    # test = dataset[0]
    # print(test.x.size())
    # print(test.y.size())
    # print(test.edge_index.size())
    # print(test.edge_attr.size())

    print("-----------")
    from torch_geometric.nn.conv import GCNConv, ChebConv

    conv_test = ChebConv(25, 1, 3)
    from predict_visits.baselines.simple_median import SimpleMedian

    simple_median = SimpleMedian()

    loader = torch_geometric.loader.DataLoader(dataset, batch_size=2)
    # print(loader.len())
    print(len(loader.dataset))

    # counter = 0
    # for test in loader:
    #     print(test.x.size())
    #     print(test.y.size())
    #     print(test.edge_index.size())
    #     print(test.edge_attr.size())
    #     # print(test.edge_index)
    #     # print(test.edge_attr)
    #     print("BATCH", test.batch)
    #     out = conv_test(test.x, test.edge_index, edge_weight=test.edge_attr)
    #     print("OUT", out.size())
    #     med = simple_median(test)
    #     print("median", med)
    #     counter += 1
    #     if counter > 3:
    #         break
