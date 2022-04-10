import os
import torch
import argparse
import json
import time
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader
import torch.nn as nn

from predict_visits.dataset import MobilityGraphDataset
from predict_visits.privacy_tools import PrivacyDataset, evaluate_privacy
from predict_visits.baselines.simple_median import SimpleMedian
from single_model_eval import evaluate
from predict_visits.model.model_wrapper import ModelWrapper
from predict_visits.utils import load_model

from predict_visits.config import model_dict

# model name is desired one
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", default="transformer", type=str, help="model type"
)
parser.add_argument("-l", "--log_labels", default=1, type=int, help="apply log")
parser.add_argument("-k", "--nr_keep", default=60, type=int, help="graph size")
parser.add_argument("-r", "--learning_rate", default=1e-4, type=float)
parser.add_argument("-e", "--nr_epochs", default=1000, type=int)
parser.add_argument("-f", "--relative_feats", type=int, default=1)
parser.add_argument("-b", "--batch_size", default=8, type=int)
parser.add_argument("-c", "--label_cutoff", default=0.95, type=float)
parser.add_argument("-z", "--embedding", default="simple", type=str)
parser.add_argument("-i", "--historic_input", default=10, type=int)
parser.add_argument("-p", "--sampling", default="normal", type=str)
parser.add_argument("-s", "--save_name", default="model", type=str)
parser.add_argument("--feature_embedding", default=0, type=int)
parser.add_argument("--include_dist", default=0, type=int)
parser.add_argument("--early_stopping", default=np.inf, type=float)
parser.add_argument("--predict_variable", default="visits", type=str)
args = parser.parse_args()

# define config
cfg = vars(args)
# model-specific args
model_cfg = model_dict[args.model]["model_cfg"]
model_cfg["historic_input"] = cfg["historic_input"]
cfg["model_cfg"] = model_cfg
print("config:", cfg)

# get basic classes / functions
NeuralModel = model_dict[args.model]["model_class"]
model_name = args.save_name


# privacy specific arguments:
print("Prediction task:", cfg["predict_variable"])
if cfg["predict_variable"] == "visits":
    dataset_class = MobilityGraphDataset
else:
    dataset_class = PrivacyDataset

# 1bin_2 files
# cfg["root"] = "data/1bin_2"
# train_data_files = [
#     "t120_1bin_2_gc1.pkl",
#     "t120_1bin_2_yumuv_graph_rep.pkl",
# "t120_1bin_tist_toph1000.pkl",
# "t120_1bin_geolife.pkl",
# ]
# test_data_files = ["t120_1bin_2_gc2.pkl"]

# normal files
train_data_files = [
    "t120_yumuv_graph_rep_poi.pkl",
    "t120_gc1_poi.pkl"
    #     "t120_tist_toph100.pkl",
    #     "t120_tist_random100.pkl",
]
test_data_files = train_data_files
# ["t120_yumuv_graph_rep_poi.pkl", "t120_gc1_poi.pkl"]

learning_rate = args.learning_rate
nr_epochs = args.nr_epochs
batch_size = args.batch_size
evaluate_every = 1

# use cuda:
if torch.cuda.is_available():
    print("CUDA available!")
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("CUDA not available!")

# set up outpath
out_path = os.path.join("trained_models", model_name)
os.makedirs("trained_models", exist_ok=True)
os.makedirs(out_path, exist_ok=True)

train_data = dataset_class(train_data_files, mode="train", device=device, **cfg)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

test_data = dataset_class(test_data_files, mode="test", device=device, **cfg)
print("num feats", train_data.num_feats)

val_data = dataset_class(test_data_files, mode="val", device=device, **cfg)

# Create model - input dimension is the number of features
model = ModelWrapper(train_data.num_feats, NeuralModel, **cfg).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total params", total_params)
cfg["nr_params"] = total_params

r3 = lambda x: round(x * 100, 2)  # round function for printing

best_test_loss, last_updated_best = np.inf, 0
start_training_time = time.time()
train_losses, test_losses, bl_losses = [], [], []
loss_fun = nn.CrossEntropyLoss()  # nn.MSELoss()
for epoch in range(nr_epochs):
    epoch_loss = 0
    # TRAIN
    for i, data_geometric in enumerate(train_loader):
        # label is part of data.y --> visits to the new location
        lab = data_geometric.y[:, -1].clone().long()

        optimizer.zero_grad()
        out = model(data_geometric.to(device))

        # MSE
        loss = loss_fun(out, lab)
        loss.backward()
        # print(torch.sum(model.dec_hidden_2.weight.grad))
        # print(torch.sum(model.dec_out.weight.grad))
        optimizer.step()

        epoch_loss += loss.item()
    # print(" out example", round(out.item(), 3), round(lab.item(), 3))
    epoch_loss = epoch_loss / train_data.len()  # compute average

    # EVALUATE
    if epoch % evaluate_every == 0:
        with torch.no_grad():
            # res_models = evaluate(
            #     {"trained": model, "median": SimpleMedian()},
            #     test_data,
            # )
            # test_loss = res_models["trained"]
            test_loss = evaluate_privacy(
                model, val_data, loss_fun, task=cfg["predict_variable"]
            )

        print(epoch, r3(epoch_loss), round(test_loss, 2))
        train_losses.append(epoch_loss)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            torch.save(model.state_dict(), os.path.join(out_path, "best_model"))
            print("saved new best model")
            best_test_loss = test_loss
            last_updated_best = epoch

    if epoch - last_updated_best > args.early_stopping:
        print("Early stopping")
        break

print("Finished training", time.time() - start_training_time)

# save model
torch.save(model.state_dict(), os.path.join(out_path, "model"))

# save config
cfg["nr_features"] = train_data.num_feats
with open(os.path.join(out_path, "cfg.json"), "w") as outfile:
    json.dump(cfg, outfile)

# save results
res = {}
res["train_losses"] = train_losses
res["test_losses"] = test_losses
res["bl_losses"] = bl_losses
res["training_time"] = time.time() - start_training_time
with open(os.path.join(out_path, "res.json"), "w") as outfile:
    json.dump(res, outfile)

# evaluate best model
model, cfg = load_model(model_name, use_best=True)
res_df_test = evaluate_privacy(
    model, test_data, loss_fun, cfg["predict_variable"], return_mode="df"
)
res_df_test["train"] = "test"
res_df_train = evaluate_privacy(
    model, train_data, loss_fun, cfg["predict_variable"], return_mode="df"
)
res_df_train["train"] = "train"
res_df = pd.concat((res_df_train, res_df_test))
os.makedirs(os.path.join("outputs", "privacy"), exist_ok=True)
res_df.to_csv(os.path.join("outputs", "privacy", model_name + ".csv"))
