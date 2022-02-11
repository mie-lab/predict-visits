import pickle
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from predict_visits.model import ClassificationModel
from predict_visits.dataset import MobilityGraphDataset

from predict_visits.baselines.simple_median import SimpleMedian
from predict_visits.baselines.knn import KNN


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


def evaluate(models_to_evaluate, test_data_loader):
    results_by_model = defaultdict(list)
    for i, (adj, node_feat, loc_feat, lab) in enumerate(test_data_loader):
        # predict the label = indegree (need to process one after the other)
        # print("------", i, lab)
        for model_name, eval_model in models_to_evaluate.items():
            out = eval_model(node_feat.float(), adj, loc_feat.float())
            test_loss = torch.sum((out - lab) ** 2).item()
            # print(model_name, out, test_loss)
            results_by_model[model_name].append(test_loss)
    return results_by_model


if __name__ == "__main__":
    import seaborn as sns

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
        required=True,
        help="Path to test data to evaluate",
    )
    args = parser.parse_args()

    model_name = args.model_name
    test_data_path = args.data_path

    # outputs directory
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", model_name)
    os.makedirs(out_path, exist_ok=True)

    # TODO: later use this script with model_name separately
    # if model_name == "knn":
    #     model = KNN(1, weighted=False)
    # elif model_name == "simple_avg":
    #     model = SimpleMedian()
    # else:
    model_checkpoint = torch.load(
        os.path.join("trained_models", model_name, "model")
    )
    graph_feat_dim = model_checkpoint["gc1.weight"].size()[0]
    temp = model_checkpoint["dec_hidden_1.weight"]
    loc_feat_dim = temp.size()[1] - temp.size()[0]
    model = ClassificationModel(
        graph_feat_dim=graph_feat_dim, loc_feat_dim=loc_feat_dim
    )
    model.load_state_dict(model_checkpoint)
    model.eval()

    # init baselines
    models_to_evaluate = {
        "Ours": model,
        "knn_1": KNN(1, weighted=False),
        "knn_5": KNN(5, weighted=False),
        "knn_5_weighted": KNN(5, weighted=True),
        "simple_median": SimpleMedian(),
    }

    # just for us for comparison
    test_dataset = MobilityGraphDataset([test_data_path])
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    # Evaluate
    results_by_model = evaluate(models_to_evaluate, test_data_loader)

    print("RESUTS")
    for model_name, losses in results_by_model.items():
        print(model_name, round(np.mean(losses), 5))

    df = pd.DataFrame(results_by_model)
    melted_df = df.melt()
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="variable", y="value", data=melted_df)
    plt.title("Loss by model")
    plt.ylim(-0.03, 0.3)
    plt.ylabel("Loss per sample")
    plt.xlabel("Model")
    plt.savefig(os.path.join(out_path, f"results.png"))

    # TODO: manual preprocessing to evaluate on fictionary locations

    # # load data
    # with open(test_data_path, "rb") as infile:
    #     (users, adjacency_graphs, node_feat_list) = pickle.load(infile)

    # # preprocess graphs manually
    # node_feats, adjacency, stats = MobilityGraphDataset.graph_preprocessing(
    #     adjacency_graphs,
    #     node_feat_list,
    #     quantile_lab=0.95,
    #     nr_keep=50,
    # )
    #
    # for i in range(len(node_feats)):
    #     # preprocess nodes and make grid for each graph
    #     processed_adj = MobilityGraphDataset.adjacency_preprocessing(
    #         adjacency[i]
    #     )

    #     # numpy to torch:
    #     inp_graph_nodes = torch.from_numpy(node_feats[i]).float()
    #     inp_adj = processed_adj.float()
    #     # Visualization:
    #     visualize_grid(node_feats[i], inp_adj, inp_graph_nodes)
