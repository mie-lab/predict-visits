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
from predict_visits.utils import get_label, load_model


def node_sampling(
    node_feats_raw, nr_trials=1, min_label=1, max_label=10, dist_thresh=500
):
    """
    Select test node from the (unprocessed) graph for the analysis
    # TODO: pass dist_thresh argument (currently hard coded)
    """
    node_feats_raw["artificial_index"] = np.arange(len(node_feats_raw))
    # the test node must fulfill some conditions
    conditions = (
        (node_feats_raw["distance"] < dist_thresh * 1000)
        & (node_feats_raw["out_degree"] <= max_label)
        & (node_feats_raw["out_degree"] >= min_label)
    )
    eligible_rows = node_feats_raw[conditions]

    # sample with prob such that label values have same prob to appear
    nr_visit_col = eligible_rows["out_degree"].values
    uni, counts = np.unique(nr_visit_col, return_counts=True)
    assert len(uni) <= (max_label - min_label + 1)
    prob_per_count = {uni[i]: 1 / counts[i] for i in range(len(uni))}
    probs = np.array([prob_per_count[l] for l in nr_visit_col])

    # to artificially include always the labels >10:
    # probs[nr_visit_col > 10] = 1000

    probs = probs / np.sum(probs)
    # sample
    test_node_indices = np.random.choice(
        eligible_rows["artificial_index"].values,
        size=min([nr_trials, len(eligible_rows)]),
        p=probs,
        replace=False,
    )
    return test_node_indices


def select_node(node_feats_raw, adjacency_raw, take_out_ind):
    """
    Cut out the test node from the (unprocessed) graph for the analysis
    """

    # the nodes that are kept for the historic mobility input (the graph):
    use_nodes = np.delete(np.arange(len(node_feats_raw)), take_out_ind)

    # crop to used nodes
    node_feats_raw_graph = node_feats_raw.iloc[use_nodes]
    adjacency_raw_graph = adjacency_raw[use_nodes]
    adjacency_raw_graph = adjacency_raw_graph[:, use_nodes]

    input_node_raw = node_feats_raw.iloc[take_out_ind : take_out_ind + 1]
    gt_label = input_node_raw["out_degree"].values[0]

    return node_feats_raw_graph, adjacency_raw_graph, input_node_raw, gt_label


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
    NR_TRIALS = 10
    MIN_LABEL = 0
    MAX_LABEL = 10

    # outputs directory
    os.makedirs("outputs", exist_ok=True)

    # init baselines
    models_to_evaluate = {"simple_median": SimpleMedian()}
    cfg_to_evaluate = {"simple_median": {}}

    # add trained model
    for model_name in os.listdir(os.path.join("trained_models", model_path)):
        if model_name[0] == ".":
            continue
        model, cfg = load_model(os.path.join(model_path, model_name))
        models_to_evaluate[model_name] = model
        cfg_to_evaluate[model_name] = cfg

        # add knn baselines with these cfgs
        models_to_evaluate["knn_3_" + model_name] = KNN(3, weighted=False)
        cfg_to_evaluate["knn_3_" + model_name] = cfg
        models_to_evaluate["knn_5_" + model_name] = KNN(5, weighted=False)
        cfg_to_evaluate["knn_5_" + model_name] = cfg
        # TODO: knn_5_weighted": KNN(5, weighted=True),
    print("Evaluating models: ", models_to_evaluate.keys())

    # # --------- Evaluation -----------
    with open(os.path.join("data", test_data_path), "rb") as infile:
        (users, adjacency_graphs, node_feat_list) = pickle.load(infile)

    results = []

    for i in range(len(users)):
        # current graph and features
        node_feats_raw = node_feat_list[i]
        adjacency_raw = adjacency_graphs[i]

        test_nodes = node_sampling(
            node_feats_raw,
            nr_trials=NR_TRIALS,
            min_label=MIN_LABEL,
            max_label=MAX_LABEL,
        )
        for take_out_ind in test_nodes:
            # select test node - get features and label
            (
                node_feats_raw_graph,
                adjacency_raw_graph,
                input_node_raw,
                gt_label,
            ) = select_node(node_feats_raw, adjacency_raw, take_out_ind)

            results_by_model = {}
            for model_name, eval_model in models_to_evaluate.items():
                # get config for preprocessing
                cfg = cfg_to_evaluate[model_name]

                # preprocess graphs manually
                (
                    node_feat,
                    adj,
                    stats_and_cutoff,
                ) = MobilityGraphDataset.graph_preprocessing(
                    adjacency_raw_graph, node_feats_raw_graph, **cfg
                )

                # preprocess labels and upper bound on labels
                label_cutoff = stats_and_cutoff[1]
                normed_label = MobilityGraphDataset.norm_label(
                    gt_label, label_cutoff, cfg.get("log_labels", False)
                )
                # this can happen (and is not avoidable) since we sample the
                # new locations from all available locations
                if normed_label > 1:
                    continue
                # print(model_name, label_cutoff, gt_label, normed_label)
                # preprocess the left out node:
                input_node, _ = MobilityGraphDataset.node_feature_preprocessing(
                    input_node_raw,
                    embedding=cfg.get("embedding", "simple"),
                    stats=stats_and_cutoff[0],
                )
                assert input_node.shape[0] == 1
                predict_node_feats = np.concatenate(
                    (input_node[0], np.array([normed_label]))
                )

                data = MobilityGraphDataset.transform_to_torch(
                    adj,
                    node_feat,
                    predict_node_feats,
                    cfg.get("relative_feats", False),
                    cfg.get("adj_is_unweighted", True),
                    cfg.get("adj_is_symmetric", True),
                    add_batch=True,
                )

                # RUN MODEL
                lab = data.y[:, -1]
                pred = eval_model(data)

                loss = torch.sum((pred - lab) ** 2).item()

                unnormed_pred = MobilityGraphDataset.unnorm_label(
                    pred.item(),
                    label_cutoff,
                    cfg.get("log_labels", False),
                )

                error = np.abs(unnormed_pred - gt_label)

                model_res = {}
                model_res["lab"] = lab.item()
                model_res["pred"] = pred.item()
                model_res["raw_pred"] = unnormed_pred
                model_res["loss"] = loss
                model_res["error"] = error
                results_by_model[model_name] = model_res

            # add trial to results
            results.append([users[i], gt_label, results_by_model])
        # Visualization:
        # visualize_grid(node_feats[i], inp_adj, inp_graph_nodes)

    with open(
        os.path.join("outputs", f"results_{args.out_name}.json"), "w"
    ) as outfile:
        json.dump(results, outfile)
