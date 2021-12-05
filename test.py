import pickle
import os
import trackintel as ti
import matplotlib.pyplot as plt
import numpy as np
import torch
from predict_visits.model import ClassificationModel
from predict_visits.dataset import MobilityGraphDataset

from scripts.graph_preprocessing import _load_graphs, graph_preprocessing


def regular_grid(node_feats, graph_list):
    test_nodes = []
    for i_graph in range(len(node_feats)):
        # we need the home node coordinates to get the distances
        all_degrees = np.array(graph_list[i_graph].out_degree())
        home_node = all_degrees[np.argmax(all_degrees[:, 1]), 0]
        home_center = graph_list[i_graph].nodes[home_node]["center"]

        # get min max max extent in both directions
        # min_x, max_x = np.min(node_feats[i_graph][:, 1]), np.max(
        #     node_feats[i_graph][:, 1]
        # )
        # min_y, max_y = np.min(node_feats[i_graph][:, 2]), np.max(
        #     node_feats[i_graph][:, 2]
        # )
        # Get quantile in both directions:
        # min_x, max_x = np.quantile(node_feats[i_graph][:, 1], 0.1), np.quantile(
        #     node_feats[i_graph][:, 1], 0.9
        # )
        # min_y, max_y = np.quantile(node_feats[i_graph][:, 2], 0.1), np.quantile(
        #     node_feats[i_graph][:, 2], 0.9
        # )
        # Get the home node +- std
        std_x, std_y = (
            np.std(node_feats[i_graph][:, 1]),
            np.std(node_feats[i_graph][:, 2]),
        )
        factor_std = 0.5
        min_x, max_x = (-1 * factor_std * std_x, factor_std * std_x)
        min_y, max_y = (-1 * factor_std * std_y, factor_std * std_y)
        x_range = np.linspace(min_x, max_x, 20)
        y_range = np.linspace(min_y, max_y, 20)

        # create feature array
        test_node_arr = []
        for x in x_range:
            for y in y_range:
                center_x, center_y = (x + home_center.x, y + home_center.y)
                dist = ti.geogr.point_distances.haversine_dist(
                    center_x, center_y, home_center.x, home_center.y
                )[0]
                test_node_arr.append([dist, x, y, 1])

        test_node_arr = np.array(test_node_arr)
        test_nodes.append(test_node_arr)
    return test_nodes


if __name__ == "__main__":
    model_path = "trained_models/first_try"
    test_data_path = "data/test_data.pkl"
    # outputs directory
    os.makedirs("outputs", exist_ok=True)

    model = ClassificationModel(input_feat_dim=4, second_input_dim=3)
    model.load_state_dict(torch.load(model_path))

    # load data
    with open(test_data_path, "rb") as infile:
        (users, adjacency_raw, node_feats) = pickle.load(infile)

    graph_list, users_study = _load_graphs("geolife")
    assert len(graph_list) == len(users)

    # preprocess nodes
    (
        processed_feats,
        dist_stats,
    ) = MobilityGraphDataset.node_feature_preprocessing(node_feats)

    # preprocess adjacencies
    processed_adj = []
    for adj in adjacency_raw:
        processed_adj.append(MobilityGraphDataset.adjacency_preprocessing(adj))

    # get locations in a grid --> these are already relative to home node!
    grid_locations = regular_grid(node_feats, graph_list)
    # preprocess the grid node features
    test_processed_feats = MobilityGraphDataset.node_feature_preprocessing(
        grid_locations, dist_stats
    )

    # Pass through model
    model.eval()
    for i in range(10):  # len(test_processed_feats)):
        # numpy to torch:
        inp_graph_nodes = torch.from_numpy(processed_feats[i]).float()
        inp_test_nodes = torch.from_numpy(
            test_processed_feats[i][:, :-1]
        ).float()
        inp_adj = processed_adj[i].float()

        # need to process one after the other
        labels_for_graph = []
        for k in range(len(test_processed_feats[i])):
            lab = model(inp_graph_nodes, inp_adj, inp_test_nodes[k])
            labels_for_graph.append(lab.item())

        # ------ Visualization ------------
        labels_for_graph = np.array(labels_for_graph)
        labels_for_graph_normed = (
            labels_for_graph - np.min(labels_for_graph)
        ) / (np.max(labels_for_graph) - np.min(labels_for_graph)) + 0.1
        plt.figure(figsize=(20, 10))
        plt.scatter(
            node_feats[i][:, 1],
            node_feats[i][:, 2],
            s=processed_feats[i][:, 3] * 100,
        )
        plt.scatter(
            grid_locations[i][:, 1],
            grid_locations[i][:, 2],
            s=labels_for_graph_normed * 100,
        )
        plt.xlim(
            np.min(grid_locations[i][:, 1]),
            np.max(grid_locations[i][:, 1]),
        )
        plt.ylim(
            np.min(grid_locations[i][:, 2]),
            np.max(grid_locations[i][:, 2]),
        )
        plt.savefig(os.path.join("outputs", f"grid_pred_{i}.png"))
