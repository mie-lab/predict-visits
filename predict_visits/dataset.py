import torch
import pickle
import os
import scipy.sparse as sp
import numpy as np


class MobilityGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self, path=os.path.join("data", "dataset.pkl"), ratio_predict=0.1
    ):
        """
        Data Loader for mobility graphs
        """
        with open(path, "rb") as infile:
            (self.users, adjacency_raw, node_feats) = pickle.load(infile)

        node_feats = self.node_feature_preprocessing(node_feats)

        # leave some out for prediction task
        self.adjacency, self.known_node_feats = [], []
        self.predict_node_feats, self.labels = [], []
        for i in range(len(self.users)):
            # we use ratio_predict percent of the nodes to predict
            num_nodes_this_graph = len(node_feats[i])
            number_nodes_predict = int(num_nodes_this_graph * ratio_predict)
            rand_perm = np.random.permutation(num_nodes_this_graph)
            # divide in known nodes and the ones to predict:
            predict_nodes = rand_perm[:number_nodes_predict]
            known_nodes = rand_perm[number_nodes_predict:]
            # reduce node feats to known nodes
            self.known_node_feats.append(node_feats[i][known_nodes])
            self.predict_node_feats.append(node_feats[i][predict_nodes, :-1])
            self.labels.append(node_feats[i][predict_nodes, -1])
            # print(
            #     "appended predict node feats", self.predict_node_feats[-1].shape
            # )
            # print("appended label", self.labels[-1])
            # print("appended known node feats", self.known_node_feats[-1].shape)
            # reduce adjacency matrix to known nodes
            adj = adjacency_raw[i][known_nodes]
            adj = adj[:, known_nodes]
            # print("adj shape prev", adjacency_raw[i].shape)
            # print("remove nodes", len(predict_nodes))
            # print("adj shape after", adj.shape)
            self.adjacency.append(self.adjacency_preprocessing(adj).float())

        self.avail_predict_nodes = [len(arr) for arr in self.labels]

        self.nr_graphs = len(self.adjacency)

        with open(path, "rb") as infile:
            (self.users, adjacency_raw, node_feats) = pickle.load(infile)

    @staticmethod
    def adjacency_preprocessing(adjacency_matrix):
        return sparse_mx_to_torch_sparse_tensor(
            preprocess_adj(adjacency_matrix)
        )

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

    @staticmethod
    def node_feature_preprocessing(node_feats):
        # normalize node feats
        for i in range(len(node_feats)):
            normed_node_feats = node_feats[i].copy()
            # normalize degree label by maximum --> we want the fraction of
            # activity of this user
            normed_node_feats[:, -1] = (
                normed_node_feats[:, -1]
                / np.sum(normed_node_feats[:, -1])
                * 100
            )
            assert np.isclose(np.sum(normed_node_feats[:, -1]), 100)
            # normed_node_feats[:, :-1] = normed_node_feats[:, :-1] / np.median(
            #     np.absolute(normed_node_feats[:, :-1]), axis=0
            # )
            # TODO: normalization
            # use log on distance
            normed_node_feats[:, 0] = np.log(normed_node_feats[:, 0] + 1)
            # normalize the distances also per user --> we get a relation to
            # the user's own perceived distance
            assert node_feats[i].shape == normed_node_feats.shape
            node_feats[i] = normed_node_feats
        return node_feats


def normalize_adj(adj):
    """From tkipf: Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """From tkipf: Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """From tkipf: Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == "__main__":
    data = MobilityGraphDataset()
    counter = 0
    for (adj, nf, pnf, lab) in data:
        print(np.max(pnf, axis=0), lab)
        # print(adj.size(), nf.shape, pnf.shape, lab)
        counter += 1
        if counter > 3:
            break
