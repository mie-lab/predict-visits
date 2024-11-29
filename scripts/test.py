import os
import json
import argparse
import numpy as np
import pickle
from collections import defaultdict
import torch

from predict_visits.dataset import MobilityGraphDataset
from predict_visits.baselines.simple_median import SimpleMedian
from predict_visits.baselines.knn import KNN
from predict_visits.utils import load_model, get_visits
from predict_visits.config import model_dict


def node_sampling(
    node_feats_raw,
    nr_trials=1,
    min_label=1,
    max_label=10,
    dist_thresh=500,
    exclude_days=7,
):
    """
    Select test node from the (unprocessed) graph for the analysis
    # TODO: pass dist_thresh argument (currently hard coded)
    """
    node_feats_raw["artificial_index"] = np.arange(len(node_feats_raw))
    # the test node must fulfill some conditions
    raw_labels = get_visits(node_feats_raw)
    conditions = (
        (node_feats_raw["distance"] < dist_thresh * 1000)
        & (raw_labels <= max_label)
        & (raw_labels >= min_label)
        # & (node_feats_raw["occured_after_days"] > exclude_days)
    )
    eligible_rows = node_feats_raw[conditions]

    # sample with prob such that label values have same prob to appear
    nr_visit_col = get_visits(eligible_rows)
    possible_nodes_col = eligible_rows["artificial_index"].values
    test_node_indices = MobilityGraphDataset.node_sampling(
        nr_visit_col,
        possible_nodes_col,
        nr_sample=nr_trials,
        sampling="balanced",
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
    gt_label = get_visits(input_node_raw)[0]

    return node_feats_raw_graph, adjacency_raw_graph, input_node_raw, gt_label


def visit_entropy(visit_numbers, cutoff=10):
    uni, counts = np.unique(visit_numbers[visit_numbers <= cutoff], return_counts=True)
    probs = counts / np.sum(counts)
    entropy = -1 * np.sum([p * np.log2(p) for p in probs])
    return entropy


def compute_dist_locs(input_node_feats, historic_node_feats):
    coords_hist = np.array(historic_node_feats[["x_normed", "y_normed"]])
    coords_input = np.array(input_node_feats[["x_normed", "y_normed"]])
    dist = np.sqrt(np.sum((coords_hist - coords_input) ** 2, 1))
    median_dist = np.median(dist)
    return median_dist


def compute_diff_locs(input_node_feats, historic_node_feats):
    dist = np.sqrt(np.sum((historic_node_feats - input_node_feats) ** 2, 1))
    median_dist = np.median(dist)
    return median_dist


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
    cfg_to_evaluate = {"simple_median": {"include_poi": False}}

    # add trained model
    for model_name in os.listdir(os.path.join("trained_models", model_path)):
        if model_name[0] == ".":
            continue
        model, cfg = load_model(os.path.join(model_path, model_name))
        models_to_evaluate[model_name] = model
        cfg_to_evaluate[model_name] = cfg

    # # KNNs for best model comparison
    cfg_knn = cfg.copy()  # use last cfg of the models
    models_to_evaluate["knn_5"] = KNN(5, weighted=False)
    cfg_to_evaluate["knn_5"] = cfg_knn
    models_to_evaluate["knn_25"] = KNN(25, weighted=False)
    cfg_to_evaluate["knn_25"] = cfg_knn

    # KNN EVAL:
    # for i in np.arange(1, 28, 2):
    #     # add knn baselines with these cfgs
    #     models_to_evaluate[f"knn_{i}"] = KNN(i, weighted=False)
    #     cfg_to_evaluate[f"knn_{i}"] = cfg
    #     models_to_evaluate[f"knn_{i}_w"] = KNN(i, weighted=True)
    #     cfg_to_evaluate[f"knn_{i}_w"] = cfg

    print("Evaluating models: ", models_to_evaluate.keys())

    # # --------- Evaluation -----------
    with open(os.path.join("data", test_data_path), "rb") as infile:
        (users, adjacency_graphs, node_feat_list) = pickle.load(infile)

    results = []

    for i in range(len(users)):
        # current graph and features
        node_feats_raw = node_feat_list[i]
        adjacency_raw = adjacency_graphs[i]

        user_entropy = visit_entropy(get_visits(node_feats_raw), cutoff=MAX_LABEL)

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

            # compute difference of sampled node from others:
            dist_from_locs = compute_dist_locs(input_node_raw, node_feats_raw)

            results_by_model = {}
            for model_name, eval_model in models_to_evaluate.items():
                # get config for preprocessing
                cfg = cfg_to_evaluate[model_name]

                # preprocess graphs manually
                (
                    node_feat,
                    adj,
                    stats_and_cutoff,
                ) = MobilityGraphDataset.graph_preprocessing(adjacency_raw_graph, node_feats_raw_graph, **cfg)
                # get labels back to positive values
                node_feat[:, -1] = np.abs(node_feat[:, -1])

                # preprocess labels and upper bound on labels
                label_cutoff = stats_and_cutoff[1]
                normed_label = MobilityGraphDataset.norm_label(gt_label, label_cutoff, cfg.get("log_labels", False))
                # this can happen (and is not avoidable) since we sample the
                # new locations from all available locations
                if normed_label > 1:
                    continue
                # print(model_name, label_cutoff, gt_label, normed_label)
                # preprocess the left out node:
                input_node, _ = MobilityGraphDataset.node_feature_preprocessing(
                    input_node_raw, stats=stats_and_cutoff[0], **cfg
                )
                diff_from_locs = compute_diff_locs(input_node, node_feat[:, :-1])

                assert input_node.shape[0] == 1
                predict_node_feats = np.concatenate((input_node[0], np.array([0])))

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
                pred = eval_model(data)

                loss = torch.sum((pred - normed_label) ** 2).item()

                unnormed_pred = MobilityGraphDataset.unnorm_label(
                    pred.item(),
                    label_cutoff,
                    cfg.get("log_labels", False),
                )
                # print(model_name)
                # print("lab", gt_label, normed_label)
                # print("pred", unnormed_pred, pred)
                # print()

                error = np.abs(unnormed_pred - gt_label)

                model_res = {}
                model_res["lab"] = normed_label
                model_res["pred"] = pred.item()
                model_res["raw_pred"] = unnormed_pred
                model_res["loss"] = loss
                model_res["error"] = error
                results_by_model[model_name] = model_res

            # add trial to results
            results.append(
                [
                    users[i],
                    gt_label,
                    results_by_model,
                    user_entropy,
                    dist_from_locs,
                    diff_from_locs,
                ]
            )
        # Visualization:
        # visualize_grid(node_feats[i], inp_adj, inp_graph_nodes)

    with open(os.path.join("outputs", f"results_{args.out_name}.json"), "w") as outfile:
        json.dump(results, outfile)
