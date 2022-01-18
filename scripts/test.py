import pickle
import os
import argparse
import trackintel as ti
import matplotlib.pyplot as plt
import numpy as np
import torch
from predict_visits.model import ClassificationModel
from predict_visits.dataset import MobilityGraphDataset


def regular_grid(coordinates):

    # get min max max extent in both directions
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
        help="Name of model (must be saved in trained_models dir)",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="data/test_data_22.pkl",
        help="Path to test data to evaluate",
    )
    args = parser.parse_args()

    model_name = args.model_name
    model_path = os.path.join("trained_models", model_name)
    test_data_path = args.data_path
    # outputs directory
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", model_name)
    os.makedirs(out_path, exist_ok=True)

    model = ClassificationModel(graph_feat_dim=3, loc_feat_dim=2)
    model.load_state_dict(torch.load(model_path))

    # load data
    with open(test_data_path, "rb") as infile:
        (users, adjacency_graphs, node_feat_list) = pickle.load(infile)

    # preprocess graphs
    node_feats, adjacency, stats = MobilityGraphDataset.graph_preprocessing(
        adjacency_graphs,
        node_feat_list,
        quantile_lab=0.95,
        nr_keep=50,
    )

    # preprocess nodes and make grid for each graph
    processed_adj, grid_locations = [], []
    for i_graph in range(len(node_feats)):
        # home node was determined beforehand in graph processing function
        # it is now the point where the displacement in x dir is zero
        home_node = np.argmin(np.abs(node_feats[i_graph][:, 0]))

        # preprocess adjacencies
        processed_adj.append(
            MobilityGraphDataset.adjacency_preprocessing(adjacency[i_graph])
        )

        # get locations in a grid --> these are already relative to home node!
        grid_locations.append(regular_grid(node_feats[i_graph]))
        # TODO: probably later we should lay out the grid locations based on the
        # unprocessed coordinates, and then use the node preprocessing method:
        # MobilityGraphDataset.node_feature_preprocessing(
        #     home_node, grid_locations, stats[i_graph]
        # )

    # Pass through model
    model.eval()
    for i in range(10):  # len(test_processed_feats)):
        # numpy to torch:
        inp_graph_nodes = torch.from_numpy(node_feats[i]).float()
        inp_test_nodes = torch.from_numpy(grid_locations[i]).float()
        inp_adj = processed_adj[i].float()

        # predict the label = indegree (need to process one after the other)
        labels_for_graph = []
        for k in range(len(grid_locations[i])):
            lab = model(inp_graph_nodes, inp_adj, inp_test_nodes[k])
            labels_for_graph.append(lab.item())

        # ------ Visualization ------------
        # normalize the predictions
        labels_for_graph = np.array(labels_for_graph)
        labels_for_graph_normed = (
            labels_for_graph - np.min(labels_for_graph)
        ) / (np.max(labels_for_graph) - np.min(labels_for_graph)) + 0.1
        plt.figure(figsize=(20, 10))
        # scatter the locations in the original user graph
        plt.scatter(
            node_feats[i][:, 0],
            node_feats[i][:, 1],
            s=node_feats[i][:, -1] * 100,
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
