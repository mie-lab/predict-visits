import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import torch

from predict_visits.model import VisitPredictionModel
from predict_visits.dataset import MobilityGraphDataset
from predict_visits.utils import get_label, load_model


def select_node(node_feats, min_label=1, max_label=10):
    """
    Select one example node from the (unprocessed) graph for the analysis
    """
    nr_visits = np.array(node_feats["out_degree"].values)
    # TODO: make sure that the probability is the same for each label value
    possible_inds = np.where(
        (nr_visits <= max_label) & (nr_visits >= min_label)
    )[0]
    chosen_ind = np.random.choice(possible_inds)
    return chosen_ind


def regular_grid(coordinates):

    # V1: get min max max extent in both directions
    # min_x, max_x = np.min(node_feats[i_graph][:, 0]), np.max(
    #     node_feats[i_graph][:, 0]
    # )
    # min_y, max_y = np.min(node_feats[i_graph][:, 1]), np.max(
    #     node_feats[i_graph][:, 1]
    # )

    # Get the home node +- std
    std_x, std_y = (
        np.std(coordinates[:, 0]),
        np.std(coordinates[:, 1]),
    )
    factor_std = 3
    min_x, max_x = (-1 * factor_std * std_x, factor_std * std_x)
    min_y, max_y = (-1 * factor_std * std_y, factor_std * std_y)
    x_range = np.linspace(min_x, max_x, 20)
    y_range = np.linspace(min_y, max_y, 20)

    # create feature array
    test_node_arr = [[x, y] for x in x_range for y in y_range]

    test_node_arr = np.array(test_node_arr)
    return test_node_arr


def visualize_grid(node_feats_inp, inp_adj, inp_graph_nodes):
    """
    TODO: adapt to new feature representation
    TODO: heatmap
    """
    grid_locations = regular_grid(node_feats_inp)
    inp_test_nodes = torch.from_numpy(grid_locations).float()

    labels_for_graph = []
    for k in range(len(grid_locations)):
        lab = model(inp_graph_nodes, inp_adj, inp_test_nodes[k])
        labels_for_graph.append(lab.item())

    # ------ Visualization ------------
    # normalize the predictions
    labels_for_graph = np.array(labels_for_graph)
    labels_for_graph_normed = (labels_for_graph - np.min(labels_for_graph)) / (
        np.max(labels_for_graph) - np.min(labels_for_graph)
    ) + 0.1
    plt.figure(figsize=(20, 10))
    # scatter the locations in the original user graph
    plt.scatter(
        node_feats_inp[:, 0],
        node_feats_inp[:, 1],
        s=node_feats_inp[:, -1] * 100,
    )
    #  scatter the grid-layout locations with point size proportional to
    # the predicted label
    plt.scatter(
        grid_locations[i][:, 0],
        grid_locations[i][:, 1],
        s=labels_for_graph_normed * 100,
    )
    plt.xlim(
        np.min(grid_locations[i][:, 0]),
        np.max(grid_locations[i][:, 0]),
    )
    plt.ylim(
        np.min(grid_locations[i][:, 1]),
        np.max(grid_locations[i][:, 1]),
    )
    plt.savefig(os.path.join(out_path, f"grid_pred_{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Name of model (must be saved in trained_models dir)",
    )
    parser.add_argument(
        "-o",
        "--out_name",
        type=str,
        required=True,
        help="Name where to save output file",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        required=True,
        help="Path to test data to evaluate",
    )
    args = parser.parse_args()

    test_data_path = args.data_path
    model_path = args.model_path
    MIN_LABEL = 1
    MAX_LABEL = 10

    model, cfg = load_model(model_path)

    with open(os.path.join("data", test_data_path), "rb") as infile:
        (users, adjacency_graphs, node_feat_list) = pickle.load(infile)

    i = 0

    node_feats_raw = node_feat_list[i]
    adjacency_raw = adjacency_graphs[i]
    take_out_ind = select_node(
        node_feats_raw, min_label=MIN_LABEL, max_label=MAX_LABEL
    )
    use_nodes = np.delete(np.arange(len(node_feats_raw)), take_out_ind)

    # crop to used nodes
    node_feats_raw_graph = node_feats_raw.iloc[use_nodes]
    adjacency_raw = adjacency_raw[use_nodes]
    adjacency_raw = adjacency_raw[:, use_nodes]

    input_node_raw = node_feats_raw.iloc[take_out_ind : take_out_ind + 1]
    gt_label = input_node_raw["out_degree"].values[0]
    # print(node_feats_raw_graph.shape, adjacency_raw.shape, input_node_raw.shape)

    # preprocess graph
    node_feats, adj, _ = MobilityGraphDataset.graph_preprocessing(
        adjacency_raw, node_feats_raw_graph, **cfg
    )

    # preprocess labels and upper bound on labels
    nr_visits_historic = get_label(adj)
    label_cutoff = MobilityGraphDataset.prep_cutoff(
        nr_visits_historic, cfg.get("label_cutoff", 0.95), cfg["log_labels"]
    )

    # preprocess the left out node:
    input_node, _ = MobilityGraphDataset.node_feature_preprocessing(
        input_node_raw, embedding=cfg.get("embedding", "simple")
    )
    # TODO: Modify the node in space or POIs!

    # create torch_geometric input data
    print(input_node.shape)
    assert input_node.shape[0] == 1
    data = MobilityGraphDataset.transform_to_torch(
        adj,
        node_feats,
        input_node[0],
        cfg["relative_feats"],
        add_batch=True,
    )

    print(
        data.x.size(),
        data.y.size(),
        data.edge_index.size(),
        data.edge_attr.size(),
        gt_label,
    )
