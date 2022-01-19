from scipy.sparse import coo
import torch
import pickle
import os
import scipy.sparse as sp
import numpy as np
from torch.functional import _return_counts, norm

from predict_visits.utils import *
from predict_visits.geo_embedding.embed import (
    std_log_embedding,
    sinusoidal_embedding,
)
from predict_visits.features.purpose import purpose_df_to_matrix


class MobilityGraphDataset(torch.utils.data.Dataset):
    """
    Dataset to yield graph and test nodes separately
    -> create new dataset if we want to do link prediction / other pipeline

    Arguments:
        ratio_predict: Ratio of nodes per graph that is left out for training
        quantile_lab: Only the nodes in the lower x-quantile are left out for
            training (e.g. home node can not be left out)
        nr_keep: graphs are reduced to the x nodes with highest degree
    """

    def __init__(
        self,
        path=os.path.join("data", "dataset.pkl"),
        ratio_predict=0.1,
        quantile_lab=0.95,
        nr_keep=50,
    ):
        """
        Data Loader for mobility graphs
        """
        self.nr_keep = nr_keep
        # Load data - Note: adjacency is a list of adjacency matrices, and
        # coordinates is a list of arrays (one for each user)
        with open(path, "rb") as infile:
            (self.users, adjacency_graphs, node_feat_list) = pickle.load(infile)
        print("Number of loaded graphs", len(self.users))

        # Preprocess the graph to get adjacency and node features
        node_feats, adjacency, stats = self.graph_preprocessing(
            adjacency_graphs,
            node_feat_list,
            quantile_lab=quantile_lab,
            nr_keep=nr_keep,
        )
        print("Number samples after preprocessing", len(node_feats))
        # store the feature dimension and normalization stats
        self.num_feats = node_feats[0].shape[1]
        self.stats = stats

        # Split into graph (adjacency and known node feats) and "new" locations
        # (predict_node_feats) with their visit numbers (labels)
        self.adjacency, self.known_node_feats = [], []
        self.predict_node_feats, self.labels = [], []
        for i in range(len(node_feats)):
            feat_vector = node_feats[i]

            # Leave out some nodes from the graph --> only take the ones that
            # are visited less frequently --> labs preprocessed such that cutoff
            # is at 1
            possible_nodes = np.where(feat_vector[:, -1] <= 1)[0]
            rand_perm = np.random.permutation(possible_nodes)
            # we use ratio_predict percent of the nodes to predict
            number_nodes_predict = int(len(feat_vector) * ratio_predict)
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

        self.nr_graphs = len(self.adjacency)

    @staticmethod
    def adjacency_preprocessing(adjacency_matrix):
        """
        Preprocess the adjacency matrix - return unweighted normalized tensor
        """
        unweighted_adjacency = (adjacency_matrix > 0).astype(int)
        return sparse_mx_to_torch_sparse_tensor(
            preprocess_adj(unweighted_adjacency)
        )

    @staticmethod
    def node_feature_preprocessing(
        node_feat_df, stats=None, embedding="simple"
    ):
        # transform geographic_coordinates
        coords_raw = np.array(node_feat_df[["x_normed", "y_normed"]])
        if embedding == "simple":
            embedded_coords, stats = std_log_embedding(coords_raw, stats)
        elif embedding == "sinus":
            embedded_coords = sinusoidal_embedding(coords_raw, 10)
        else:
            raise ValueError(
                f"Wrong embedding type {embedding}, must be simple or sinus"
            )
        # purpose feature
        purpose_feature_arr = purpose_df_to_matrix(node_feat_df)
        # TODO: add other features
        feature_matrix = np.hstack((embedded_coords, purpose_feature_arr))
        return feature_matrix, stats

    @staticmethod
    def graph_preprocessing(
        adjacency_graphs, node_feature_dfs, quantile_lab=0.9, nr_keep=50
    ):
        """Preprocess the node features of the graph"""
        node_feats, adjacency_matrices, stats = [], [], []
        for adjacency, node_feature_df in zip(
            adjacency_graphs, node_feature_dfs
        ):

            # 1) crop or pad adjacency matrix to the x nodes with highest degree
            unweighted_adj = (adjacency > 0).astype(int)
            overall_degree = (
                np.array(np.sum(unweighted_adj, axis=0))[0]
                + np.array(np.sum(unweighted_adj, axis=1))[0]
            )
            use_nodes = np.argsort(overall_degree)[-nr_keep:]
            # crop or pad adjacency
            adj_new = crop_pad_sparse_matrix(adjacency, use_nodes, nr_keep)
            adjacency_matrices.append(adj_new)

            # 2) process node features
            # transform geographic coordinates and make feature matrix
            (
                feature_matrix,
                feat_stats,
            ) = MobilityGraphDataset.node_feature_preprocessing(node_feature_df)
            stats.append(feat_stats)
            # restrict to use_nodes and pad (to align indices with adjacency)
            feature_matrix = feature_matrix[use_nodes]
            feature_matrix = np.pad(
                feature_matrix,
                ((0, nr_keep - feature_matrix.shape[0]), (0, 0)),
            )

            # 3) Get label (number of visits)
            # get the weighted in degree (the label in the prediction task)
            label = get_label(adj_new)
            # normalize label by the quantile
            label_cutoff = max([np.quantile(label, quantile_lab), 1])
            label = label / label_cutoff

            # concatenate the other features with the labels and append
            concat_feats_labels = np.concatenate(
                (feature_matrix, np.expand_dims(label, 1)), axis=1
            )
            node_feats.append(concat_feats_labels)
        return node_feats, adjacency_matrices, stats

    def __len__(self):
        return self.nr_graphs

    def __getitem__(self, idx):
        rand_label_ind = np.random.permutation(self.avail_predict_nodes[idx])[0]
        return (
            self.adjacency[idx],
            self.known_node_feats[idx],
            self.predict_node_feats[idx][rand_label_ind],
            self.labels[idx][rand_label_ind],
        )


if __name__ == "__main__":
    data = MobilityGraphDataset("data/test_data_22.pkl")
    counter = 0
    for (adj, nf, pnf, lab) in data:
        print(nf)
        print(adj.size(), nf.shape, pnf.shape, lab)
        counter += 1
        if counter > 10:
            break
