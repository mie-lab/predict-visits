from cgi import test
import pickle
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import HomeLocationDataset, inverse_coord_normalization
from model import HomeNet


def take_avg(node_feats, mode="median"):
    wo_home = node_feats[node_feats["distance"] > 0]
    other_locs = wo_home[["x_normed", "y_normed"]]
    locs_arr = np.array(other_locs)
    if mode == "median":
        res = np.median(locs_arr, axis=0)
    elif mode == "mean":
        res = np.mean(locs_arr, axis=0)
    #     plt.scatter(locs_arr[:, 0], locs_arr[:, 1])
    #     plt.scatter(res[0], res[1])
    return np.linalg.norm(res)


def weighted_avg(node_feats):
    wo_home = node_feats[node_feats["distance"] > 0]
    weights = wo_home["in_degree"].values
    weights = weights / np.sum(weights)
    other_locs = np.array(wo_home[["x_normed", "y_normed"]])
    res = np.sum(other_locs * np.expand_dims(weights, axis=1), axis=0)
    return np.linalg.norm(res)


def top_k(node_feats, k=3, mode="mean"):
    wo_home = (node_feats[node_feats["distance"] > 0]).sort_values(
        "in_degree", ascending=False
    )
    # print(wo_home[["x_normed", "y_normed", "in_degree"]])
    other_locs = np.array(wo_home[["x_normed", "y_normed"]])[:k]
    most_visited = np.mean(other_locs, axis=0)
    if mode == "median" and k <= 3:
        print("warning: median only makes sense for k>3")
    if mode == "median" and k > 3:
        res = np.median(most_visited, axis=0)
    elif mode == "mean":
        res = np.mean(most_visited, axis=0)
    return np.linalg.norm(res)


def evaluate(model, test_loader, criterion=torch.nn.MSELoss()):
    epoch_test_loss = 0
    test_distance = []
    for j, (locs, home_node, ref_points) in enumerate(test_loader):

        inp = locs.reshape(home_node.size()[0], -1)
        y_pred = model(inp)

        test_loss = criterion(y_pred, home_node)
        epoch_test_loss += test_loss.item()

        # for test data: check real error (dist to actual home)
        to_coords = inverse_coord_normalization(y_pred).reshape(-1, 3, 2)
        absolute_coord_error = torch.mean(to_coords + ref_points, dim=1)
        distance_from_home = torch.sqrt(torch.sum(absolute_coord_error ** 2))
        test_distance.append(distance_from_home)
    return epoch_test_loss, test_distance


if __name__ == "__main__":
    study = "gc2"
    eval_model = "gc2_1"

    eval_model_path = os.path.join(
        "trained_models", "home_prediction", eval_model
    )

    # load torch and numpy dataset
    with open(
        os.path.join(eval_model_path, "cfg_res.json"),
        "r",
    ) as infile:
        cfg = json.load(infile)
    cfg["nr_ref"] = cfg["ref_points"]
    cfg["embedding"] = "none"
    coord_dim = cfg["nr_ref"] * 2
    nr_inp = (cfg["nr_keep"] - 1) * (coord_dim + cfg["nr_features"])
    test_data_files = [f"t120_{study}_poi.pkl"]
    test_dataset = HomeLocationDataset(test_data_files, **cfg)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # load model
    model_checkpoint = torch.load(
        os.path.join(eval_model_path, "model"),
        map_location=torch.device("cpu"),
    )
    model = HomeNet(nr_inp, coord_dim)
    model.load_state_dict(model_checkpoint)
    model.eval()

    # run model
    with torch.no_grad():
        _, eval_model_distances = evaluate(model, test_loader)

    # load data for baselines
    with open(
        os.path.join("data", f"t120_{study}.pkl"),
        "rb",
    ) as outfile:
        (user_id_list, adjacency_list, node_feat_list) = pickle.load(outfile)

    # run baselines
    err_avg = [
        take_avg(node_feats, mode="mean") for node_feats in node_feat_list
    ]
    err_median = [
        take_avg(node_feats, mode="median") for node_feats in node_feat_list
    ]
    err_weighted = [weighted_avg(node_feats) for node_feats in node_feat_list]
    err_top3 = [top_k(node_feats) for node_feats in node_feat_list]
    err_top5 = [top_k(node_feats, k=5) for node_feats in node_feat_list]
    err_top5_median = [
        top_k(node_feats, k=5, mode="median") for node_feats in node_feat_list
    ]

    method_names = [
        "Mean",
        "Median",
        "Weighted avg",
        "Top 3",
        "Top 5",
        "Top 5 Median",
        "Feed forward model",
    ]
    methods = [
        err_avg,
        err_median,
        err_weighted,
        err_top3,
        err_top5,
        err_top5_median,
        eval_model_distances,
    ]

    get_km = lambda x: round(x) / 1000
    for errs, name in zip(methods, method_names):
        print(get_km(np.mean(errs)), get_km(np.median(errs)), "km", name)
