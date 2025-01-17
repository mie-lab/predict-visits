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
from predict_visits.dataset import MobilityGraphDataset

from predict_visits.baselines.simple_median import SimpleMedian
from predict_visits.baselines.knn import KNN
from predict_visits.utils import load_model

from predict_visits.config import model_dict


def evaluate(models_to_evaluate, test_data, return_mode="mean"):
    """return_mode:  ["mean", "list"]"""
    nr_samples = test_data.len()
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    # save results in dict
    results_by_model = defaultdict(list)
    for _, data in enumerate(test_data_loader):
        lab = data.y[:, -1].clone()
        for model_name, eval_model in models_to_evaluate.items():
            # forward pass
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
        "knn_3": KNN(3, weighted=False),
        "knn_5": KNN(5, weighted=False),
        "knn_5_weighted": KNN(5, weighted=True),
        "simple_median": SimpleMedian(),
    }

    # add trained model
    # for model_name in os.listdir(model_path):
    #     if model_name[0] == ".":
    #         continue
    model, cfg = load_model(model_path)
    models_to_evaluate[args.out_name] = model

    # ----- Manual evaluation -----------
    with open(os.path.join("data", test_data_path), "rb") as infile:
        (users, adjacency_graphs, node_feat_list) = pickle.load(infile)

    results = []

    for i in range(len(users)):
        # preprocess graphs manually
        (
            node_feat,
            adj,
            stats_and_cutoff,
        ) = MobilityGraphDataset.graph_preprocessing(
            adjacency_graphs[i], node_feat_list[i], **cfg
        )
        label_cutoff = stats_and_cutoff[1]

        # select test node
        for k in range(nr_trials):
            (
                adj_trial,
                known_node_feats,
                predict_node_feats,
                predict_node_index,
            ) = MobilityGraphDataset.select_test_node(
                node_feat, adj, sampling="balanced"
            )
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
                # forward pass
                pred = eval_model(data)

                loss = torch.sum((pred - lab) ** 2).item()

                unnormed_pred = MobilityGraphDataset.unnorm_label(
                    pred.item(),
                    label_cutoff,
                    cfg["log_labels"],
                )
                unnormed_lab = MobilityGraphDataset.unnorm_label(
                    lab.item(),
                    label_cutoff,
                    cfg["log_labels"],
                )

                error = np.abs(unnormed_pred - unnormed_lab)

                model_res = {}
                model_res["pred"] = pred.item()
                model_res["raw_pred"] = unnormed_pred
                model_res["loss"] = loss
                model_res["error"] = error
                results_by_model[model_name] = model_res

            # add trial to results
            results.append([lab.item(), unnormed_lab, results_by_model])

        # Visualization:
        # visualize_grid(node_feats[i], inp_adj, inp_graph_nodes)

    with open(
        os.path.join("outputs", f"results_{args.out_name}.json"), "w"
    ) as outfile:
        json.dump(results, outfile)

    # ------ Simple loss evaluation -------------

    # load dataset
    test_dataset = MobilityGraphDataset([args.data_path], **cfg)

    # Evaluate
    results_by_model = evaluate(
        models_to_evaluate, test_dataset, return_mode="list"
    )

    print("RESULTS")
    for model_name, losses in results_by_model.items():
        print(model_name, round(np.mean(losses), 5))
