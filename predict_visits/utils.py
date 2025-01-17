import numpy as np
import json
import os
import torch
import scipy.sparse as sp
from predict_visits.model.model_wrapper import ModelWrapper
from predict_visits.config import model_dict


def load_model(model_path):
    with open(
        os.path.join("trained_models", model_path, "cfg.json"), "r"
    ) as infile:
        cfg = json.load(infile)
    # get info which model to use
    cfg_model = model_dict[cfg["model"]]
    NeuralModel = cfg_model["model_class"]

    # init model
    model = ModelWrapper(cfg["nr_features"], NeuralModel, **cfg)

    # load checkpoint
    model_checkpoint = torch.load(
        os.path.join("trained_models", model_path, "model"),
        map_location=torch.device("cpu"),
    )
    # such a shit: due to some pytorch version stuff or so, the checkpoint from
    # the cluster does not fit to my model. Solved with super ugly code below
    try:
        model.load_state_dict(model_checkpoint)
    except:
        new_checkpoint = {}
        for key in model_checkpoint.keys():
            ckpt_shape = model_checkpoint[key].shape
            if len(ckpt_shape) == 3:
                for i in range(ckpt_shape[0]):
                    new_key = ".".join(
                        key.split(".")[:-1]
                        + ["lins"]
                        + [str(i)]
                        + [key.split(".")[-1]]
                    )
                    # missing: swapaxes
                    new_checkpoint[new_key] = torch.swapaxes(
                        model_checkpoint[key][i], 1, 0
                    )
            else:
                new_checkpoint[key] = model_checkpoint[key]
        model.load_state_dict(new_checkpoint)

    model.eval()
    return model, cfg


def get_visits(node_feat_df):
    return np.max(node_feat_df[["in_degree", "out_degree"]].values, axis=1)


def get_home_node(adjacency):
    # find home node
    visit_nr = get_visits_adj(adjacency)
    return np.argmax(visit_nr)


def get_visits_adj(adjacency):
    # get the indegrees
    ind = np.array(np.sum(adjacency, axis=0))[0]
    # get out degrees
    outd = np.array(np.sum(adjacency, axis=1)).flatten()
    return np.max(np.vstack((ind, outd)), axis=0)


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


def crop_pad_sparse_matrix(adjacency, use_nodes, keep_nodes):
    """
    Reduce sparse matrix to the indices use_nodes or pad such that the number
    of rows and columns is keep_nodes
    """
    adj_temp = adjacency[use_nodes]
    adj_temp = adj_temp[:, use_nodes]
    # check if we are actually below and need to pad
    diff = keep_nodes - adj_temp.shape[0]
    if diff > 0:
        from scipy.sparse import vstack, hstack

        adj_temp = hstack((adj_temp, np.zeros((adj_temp.shape[0], diff))))
        adj_temp = vstack((adj_temp, np.zeros((diff, 50)))).tocsr()
    return adj_temp


def old_node_preprocessing(node_feats, dist_stats=None):
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
        # compute stats that must be saved for the evaluation
        if dist_stats is None:
            # note: dist_means is not used currently
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
        # # SANITY CHECK: can we predict just the normalized distance?
        # # Note: need to comment lab max in code above
        # normed_node_feats[:, -1] = normed_node_feats[:, 0] / dist_means[0]
        # if dist_stats is None:
        #     lab_max = np.max(normed_node_feats[:, -1])
        #     lab_max_list.append(lab_max)
        # else:
        #     lab_max = lab_max_list[i]
        # normed_node_feats[:, -1] = normed_node_feats[:, -1] / lab_max

        # 2) Normalize labels (#visits) by max because prediction should be
        # time-period independent
        normed_node_feats[:, -1] = normed_node_feats[:, -1] / lab_max
        # 3) normalize distances by ?? Median / mean and std
        normed_node_feats[:, :-1] = (normed_node_feats[:, :-1]) / dist_stds
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
