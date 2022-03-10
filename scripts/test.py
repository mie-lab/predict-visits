import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from collections import defaultdict
import torch
from torch_geometric.data import DataLoader
from predict_visits.model import VisitPredictionModel
from predict_visits.dataset import MobilityGraphDataset

from predict_visits.baselines.simple_median import SimpleMedian
from predict_visits.baselines.knn import KNN
from predict_visits.utils import get_label


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


def evaluate(models_to_evaluate, test_data, return_mode="mean"):
    """return_mode:  ["mean", "list"]"""
    nr_samples = test_data.len()
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    # save results in dict
    results_by_model = defaultdict(list)
    for _, data in enumerate(test_data_loader):
        lab = data.y[:, -1]
        for model_name, eval_model in models_to_evaluate.items():
            out = eval_model(data)
            test_loss = torch.sum((out - lab) ** 2).item()
            results_by_model[model_name].append(test_loss)
    if return_mode == "mean":
        # take average for each model
        final_res = {}
        for model_name in results_by_model.keys():
            final_res[model_name] = (
                np.sum(results_by_model[model_name]) / nr_samples
            )
        return final_res
    elif return_mode == "list":
        return results_by_model
    else:
        raise ValueError("return_mode must be one of [mean, list]")


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
    # path to folder with the model to be evaluated
    model_path = args.model_path
    nr_trials = 5

    # outputs directory
    os.makedirs("outputs", exist_ok=True)

    # init baselines
    models_to_evaluate = {
        "knn_1": KNN(1, weighted=False),
        "knn_5": KNN(5, weighted=False),
        "knn_5_weighted": KNN(5, weighted=True),
        "simple_median": SimpleMedian(),
    }

    # add trained model
    # for model_name in os.listdir(model_path):
    #     if model_name[0] == ".":
    #         continue
    model_checkpoint = torch.load(
        os.path.join("trained_models", model_path, "model"),
        map_location=torch.device("cpu"),
    )
    with open(
        os.path.join("trained_models", model_path, "cfg_res.json"), "r"
    ) as infile:
        cfg = json.load(infile)
    model = VisitPredictionModel(cfg["nr_features"])
    model.load_state_dict(model_checkpoint)
    model.eval()
    models_to_evaluate[args.out_name] = model

    # ----- Manual evaluation -----------
    with open(os.path.join("data", test_data_path), "rb") as infile:
        (users, adjacency_graphs, node_feat_list) = pickle.load(infile)

    # preprocess graphs manually
    node_feats, adjacency, stats = MobilityGraphDataset.graph_preprocessing(
        adjacency_graphs, node_feat_list, **cfg
    )

    results = []

    for i in range(len(node_feats)):
        adj = adjacency[i]
        node_feat = node_feats[i]
        # preprocess nodes and make grid for each graph
        raw_labels = get_label(adj)
        label_cutoff = max(
            [np.quantile(raw_labels, cfg.get("quantile_lab", 0.95)), 1]
        )

        # select test node
        for k in range(nr_trials):
            (
                adj_trial,
                known_node_feats,
                predict_node_feats,
                predict_node_index,
            ) = MobilityGraphDataset.select_test_node(node_feat, adj)
            # transform to pytorch geometric data
            data = MobilityGraphDataset.transform_to_torch(
                adj_trial,
                known_node_feats,
                predict_node_feats,
                cfg["relative_feats"],
                add_batch=True,
            )
            lab = data.y[:, -1]
            # only as sanity check
            # unnormed_lab = MobilityGraphDataset.unnorm_label(
            #     lab.item(),
            #     label_cutoff,
            #     cfg["log_labels"],
            # )
            results_by_model = {}
            for model_name, eval_model in models_to_evaluate.items():
                pred = eval_model(data)

                loss = torch.sum((pred - lab) ** 2).item()

                unnormed_pred = MobilityGraphDataset.unnorm_label(
                    pred.item(),
                    label_cutoff,
                    cfg["log_labels"],
                )

                error = np.abs(unnormed_pred - raw_labels[predict_node_index])

                model_res = {}
                model_res["pred"] = pred.item()
                model_res["raw_pred"] = unnormed_pred
                model_res["loss"] = loss
                model_res["error"] = error
                results_by_model[model_name] = model_res

            # add trial to results
            results.append(
                [lab.item(), raw_labels[predict_node_index], results_by_model]
            )

        # Visualization:
        # visualize_grid(node_feats[i], inp_adj, inp_graph_nodes)

    with open(
        os.path.join("outputs", f"results_{args.out_name}.json"), "w"
    ) as outfile:
        json.dump(results, outfile)

    # ------ Simple loss evaluation -------------

    # load dataset
    test_dataset = MobilityGraphDataset([test_data_path], **cfg)

    # Evaluate
    results_by_model = evaluate(
        models_to_evaluate, test_dataset, return_mode="list"
    )

    print("RESULTS")
    for model_name, losses in results_by_model.items():
        print(model_name, round(np.mean(losses), 5))

    # df = pd.DataFrame(results_by_model)
    # melted_df = df.melt()
    # plt.figure(figsize=(10, 5))
    # sns.boxplot(x="variable", y="value", data=melted_df)
    # plt.title("Loss by model")
    # plt.ylim(-0.01, 0.15)
    # plt.ylabel("Loss per sample")
    # plt.xlabel("Model")
    # plt.savefig(os.path.join(out_path, f"results.png"))
