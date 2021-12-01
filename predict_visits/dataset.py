import torch
import pickle
import os
import scipy.sparse as sp
import numpy as np
from torch.functional import _return_counts, norm


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

        # print(np.mean(np.absolute(node_feats[0]), axis=0))
        # print()
        # all_labels = [e[-1] for f in node_feats for e in f]
        # print(np.unique(all_labels, return_counts=True))

        # leave some out for prediction task
        self.adjacency, self.known_node_feats = [], []
        self.predict_node_feats, self.labels = [], []
        for i in range(len(self.users)):
            # we use ratio_predict percent of the nodes to predict
            num_nodes_this_graph = len(node_feats[i])
            number_nodes_predict = int(num_nodes_this_graph * ratio_predict)
            # TODO: exclude two nodes with highest number of visits --> outlier
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
    def node_feature_preprocessing(node_feats, dist_stats=None):
        if dist_stats is not None:
            dist_mean_list, dist_std_list, lab_max_list = dist_stats
        else:
            dist_mean_list, dist_std_list, lab_max_list = [], [], []
        # normalize node feats:
        # normalize the distances also per user --> we get a relation to
        # the user's own perceived distance
        normed_node_feats_list = []
        for i in range(len(node_feats)):
            normed_node_feats = node_feats[i].copy()
            # print("prev", normed_node_feats[:10])
            # get sign (mainly of coordinate features)
            temp_sign = np.sign(normed_node_feats)
            # 1) Apply log because distances and visits both follow power law
            normed_node_feats = temp_sign * np.log(
                np.absolute(normed_node_feats) + 1
            )
            # 2) Normalize labels (#visits) by max because prediction should be
            # time-period independent
            if dist_stats is None:
                dist_means = np.mean(normed_node_feats[:, :-1], axis=0)
                dist_stds = np.std(normed_node_feats[:, :-1], axis=0)
                lab_max = np.max(normed_node_feats[:, -1])
                dist_mean_list.append(dist_means)
                dist_std_list.append(dist_stds)
                lab_max_list.append(lab_max)
            else:
                dist_means, dist_stds, lab_max = (
                    dist_mean_list[i],
                    dist_std_list[i],
                    lab_max_list[i],
                )

            normed_node_feats[:, -1] = normed_node_feats[:, -1] / lab_max
            # 3) normalize distances by ?? Median / mean and std
            normed_node_feats[:, :-1] = (
                normed_node_feats[:, :-1] - dist_means
            ) / dist_stds
            assert node_feats[i].shape == normed_node_feats.shape
            normed_node_feats_list.append(normed_node_feats)
            # print("after", normed_node_feats[:10])
        if dist_stats is None:
            return normed_node_feats_list, (
                dist_mean_list,
                dist_std_list,
                lab_max_list,
            )
        return normed_node_feats_list


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
    data = MobilityGraphDataset("data/train_data.pkl")
    counter = 0
    for (adj, nf, pnf, lab) in data:
        print(nf, lab)
        # print(adj.size(), nf.shape, pnf.shape, lab)
        counter += 1
        if counter > 3:
            break
