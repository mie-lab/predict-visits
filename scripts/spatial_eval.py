import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import torch

from predict_visits.model.graph_resnet import VisitPredictionModel
from predict_visits.dataset import MobilityGraphDataset
from predict_visits.utils import get_label, load_model
from predict_visits.config import model_dict
from predict_visits.model.transforms import NoTransform


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
    factor_std = 1
    min_x, max_x = (-1 * factor_std * std_x, factor_std * std_x)
    min_y, max_y = (-1 * factor_std * std_y, factor_std * std_y)
    x_range = np.linspace(min_x, max_x, 20)
    y_range = np.linspace(min_y, max_y, 20)

    # create feature array
    test_node_arr = [[x, y] for x in x_range for y in y_range]

    test_node_arr = np.array(test_node_arr)
    return test_node_arr


def create_fake_nodes(node_feat_df):
    loc_grid = regular_grid(np.array(node_feat_df[["x_normed", "y_normed"]]))
    fake_nodes = pd.DataFrame(loc_grid, columns=["x_normed", "y_normed"])
    fake_nodes["distance"] = np.sqrt(
        fake_nodes["x_normed"].values ** 2 + fake_nodes["y_normed"].values ** 2
    )
    fake_nodes["purpose"] = "unknown"
    return fake_nodes


def plot_spatial_distribution_3d(
    grid_locations,
    grid_labels,
    user_locations,
    user_labels,
    max_label=10,
    save_path="outputs/plot3d.png",
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    x, y = np.swapaxes(grid_locations, 1, 0)
    z = grid_labels
    ax.plot_trisurf(x, y, z)  # scatter plot_trisurf
    x, y = np.swapaxes(user_locations, 1, 0)
    z = user_labels
    ax.scatter(x, y, z, c="red")
    plt.xlabel("X coordinate (relative to home)")
    plt.ylabel("Y coordinate (relative to home)")
    ax.set_zlabel("Number of visits")
    ax.set_zlim(0, max_label)
    plt.savefig(save_path)
    # plt.show()


def plot_spatial_distribution_2d(
    grid_locations,
    grid_labels,
    user_locations,
    user_labels,
    save_path="outputs/plot2d.png",
):

    grid_labels_normed = (grid_labels - np.min(grid_labels)) / (
        np.max(grid_labels) - np.min(grid_labels)
    ) + 0.1
    user_labels_normed = (user_labels - np.min(user_labels)) / (
        np.max(user_labels) - np.min(user_labels)
    ) + 0.1

    plt.figure(figsize=(7, 7))
    # scatter the locations in the original user graph
    plt.scatter(
        user_locations[:, 0],
        user_locations[:, 1],
        s=user_labels_normed * 100,
    )
    #  scatter the grid-layout locations with point size proportional to
    # the predicted label
    plt.scatter(
        grid_locations[:, 0],
        grid_locations[:, 1],
        s=grid_labels_normed * 100,
    )
    plt.xlim(
        np.min(grid_locations[:, 0]),
        np.max(grid_locations[:, 0]),
    )
    plt.ylim(
        np.min(grid_locations[:, 1]),
        np.max(grid_locations[:, 1]),
    )
    plt.xlabel("X coordinate (relative to home)")
    plt.ylabel("Y coordinate (relative to home)")
    plt.savefig(save_path)
    # plt.show()


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
    model.eval()
    cfg["include_poi"] = False

    # add transform function
    model_cfg = model_dict[cfg["model"]]
    transform_for_model = model_cfg.get("inp_transform", NoTransform)(**cfg)

    with open(os.path.join("data", test_data_path), "rb") as infile:
        (users, adjacency_graphs, node_feat_list) = pickle.load(infile)

    for i in range(10):

        node_feats_raw = node_feat_list[i]
        adjacency_raw = adjacency_graphs[i]

        # get fake nodes:
        fake_node_df = create_fake_nodes(node_feats_raw)

        # preprocess graph
        (
            node_feats,
            adj,
            stats_and_cutoff,
        ) = MobilityGraphDataset.graph_preprocessing(
            adjacency_raw, node_feats_raw, **cfg
        )
        label_cutoff = stats_and_cutoff[1]

        # preprocess the left out node:
        input_node, _ = MobilityGraphDataset.node_feature_preprocessing(
            fake_node_df, stats=stats_and_cutoff[0], **cfg
        )
        predict_node_feats = np.concatenate(
            (input_node, np.zeros((len(input_node), 1))), axis=1
        )

        # create torch_geometric input data
        grid_labels = []
        for predict_node in predict_node_feats:
            data = MobilityGraphDataset.transform_to_torch(
                adj,
                node_feats,
                predict_node,
                cfg["relative_feats"],
                cfg.get("adj_is_unweighted", True),
                cfg.get("adj_is_symmetric", True),
                add_batch=True,
            )

            # final model-dependent transform
            inp_data = transform_for_model(data)

            # RUN MODEL
            lab = data.y[:, -1]
            pred = model(inp_data)

            unnormed_pred = MobilityGraphDataset.unnorm_label(
                pred.item(),
                label_cutoff,
                cfg.get("log_labels", False),
            )
            grid_labels.append(unnormed_pred)

        grid_labels = np.array(grid_labels)
        grid_locations = np.array(fake_node_df[["x_normed", "y_normed"]])
        user_locations = np.array(node_feats_raw[["x_normed", "y_normed"]])
        user_labels = np.array(node_feats_raw[["out_degree"]])

        plot_spatial_distribution_2d(
            grid_locations,
            grid_labels,
            user_locations,
            user_labels,
            save_path=os.path.join(f"outputs/spatial_eval/plot_2d_{i}"),
        )

        plot_spatial_distribution_3d(
            grid_locations,
            grid_labels,
            user_locations,
            user_labels,
            save_path=os.path.join(f"outputs/spatial_eval/plot_3d_{i}"),
        )
