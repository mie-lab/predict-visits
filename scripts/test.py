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


def select_node(
    node_feats_raw, adjacency_raw, min_label=1, max_label=10, dist_thresh=500
):
    """
    Select one example node from the (unprocessed) graph for the analysis
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

    # TODO: make sure that the probability is the same for each label value
    take_out_ind = np.random.choice(eligible_rows["artificial_index"].values)

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
    NR_TRIALS = 5
    MIN_LABEL = 1
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

    # --------- Evaluation -----------
    with open(os.path.join("data", test_data_path), "rb") as infile:
        (users, adjacency_graphs, node_feat_list) = pickle.load(infile)

    results = []

    for i in range(len(users)):
        # current graph and features
        node_feats_raw = node_feat_list[i]
        adjacency_raw = adjacency_graphs[i]

        for k in range(NR_TRIALS):
            # select test node - get features and label
            (
                node_feats_raw_graph,
                adjacency_raw_graph,
                input_node_raw,
                gt_label,
            ) = select_node(
                node_feats_raw,
                adjacency_raw,
                min_label=MIN_LABEL,
                max_label=MAX_LABEL,
            )
            # raw labels are only needed for the label cutoff
            raw_labels = node_feats_raw_graph["out_degree"].values

            results_by_model = {}
            for model_name, eval_model in models_to_evaluate.items():
                # get config for preprocessing
                cfg = cfg_to_evaluate[model_name]

                # preprocess graphs manually
                (
                    node_feat,
                    adj,
                    stats,
                ) = MobilityGraphDataset.graph_preprocessing(
                    adjacency_raw_graph, node_feats_raw_graph, **cfg
                )

                # preprocess labels and upper bound on labels
                label_cutoff = MobilityGraphDataset.prep_cutoff(
                    raw_labels,
                    cfg.get("label_cutoff", 0.95),
                    cfg.get("log_labels", False),
                )
                normed_label = MobilityGraphDataset.norm_label(
                    gt_label, label_cutoff, cfg.get("log_labels", False)
                )

                # preprocess the left out node:
                input_node, _ = MobilityGraphDataset.node_feature_preprocessing(
                    input_node_raw,
                    embedding=cfg.get("embedding", "simple"),
                    stats=stats,
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
                pred = eval_model(data)
                lab = data.y[:, -1]

                loss = torch.sum((pred - lab) ** 2).item()

                unnormed_pred = MobilityGraphDataset.unnorm_label(
                    pred.item(),
                    label_cutoff,
                    cfg.get("log_labels", False),
                )

                error = np.abs(unnormed_pred - gt_label)

                model_res = {}
                model_res["pred"] = pred.item()
                model_res["raw_pred"] = unnormed_pred
                model_res["loss"] = loss
                model_res["error"] = error
                results_by_model[model_name] = model_res

            # add trial to results
            results.append([lab.item(), gt_label, results_by_model])

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
