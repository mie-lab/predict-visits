import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HomeLocationDataset, inverse_coord_normalization
from model import HomeNet
from test import evaluate

# model name is desired one
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="model",
    type=str,
    help="Name under which to save model",
)
parser.add_argument(
    "-l",
    "--log_labels",
    default=True,
    type=bool,
    help="Use logarithm on labels",
)
parser.add_argument(
    "-k",
    "--nr_keep",
    default=10,
    type=int,
    help="graph size",
)
parser.add_argument(
    "-r",
    "--learning_rate",
    default=5e-3,
    type=float,
    help="learning rate",
)
parser.add_argument(
    "-p",
    "--ref_points",
    default=3,
    type=int,
    help="number of reference points",
)
parser.add_argument(
    "-e",
    "--nr_epochs",
    default=500,
    type=int,
    help="number of epochs",
)
parser.add_argument(
    "-b",
    "--batch_size",
    default=4,
    type=int,
    help="batch size",
)
args = parser.parse_args()

# Variables:
log_labels = args.log_labels
nr_keep = args.nr_keep
nr_ref = args.ref_points
model_name = args.model

train_data_files = ["t120_gc2_poi.pkl", "t120_yumuv_graph_rep_poi.pkl"]
test_data_files = ["t120_gc1_poi.pkl"]
kwargs = {
    "root": "data",
    "embedding": "none",
    "log_labels": log_labels,
    "nr_keep": nr_keep,
}
train_dataset = HomeLocationDataset(train_data_files, nr_ref=nr_ref, **kwargs)
test_dataset = HomeLocationDataset(test_data_files, nr_ref=nr_ref, **kwargs)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# outpaths
out_path = os.path.join("trained_models", "home_prediction", model_name)
os.makedirs(os.path.join("trained_models", "home_prediction"), exist_ok=True)
os.makedirs(out_path, exist_ok=True)

# Init model and loss
coord_dim = nr_ref * 2
# flat input:
nr_inp = (nr_keep - 1) * (coord_dim + train_dataset.num_feats)
model = HomeNet(nr_inp, coord_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

start_training_time = time.time()
train_losses, test_losses = [], []
for epoch in range(args.nr_epochs):
    epoch_train_loss = 0
    for (locs, home_node, _) in train_loader:
        optimizer.zero_grad()

        inp = locs.reshape(-1, nr_inp)
        y_pred = model(inp)

        # calculating loss
        loss = criterion(y_pred, home_node)

        # backprop
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    if epoch % 5 == 0:
        # TEST
        with torch.no_grad():
            (epoch_test_loss, test_distance) = evaluate(
                model, test_loader, criterion=criterion
            )

        r_train = round(epoch_train_loss / len(train_loader) * 1000)
        r_test = round(epoch_test_loss / len(test_loader) * 1000)
        print(
            "Epoch",
            epoch,
            "Losses",
            r_train,
            r_test,
            "Distances:",
            round(np.mean(test_distance)),
            round(np.median(test_distance)),
        )

        train_losses.append(r_train)
        test_losses.append(r_test)

torch.save(model.state_dict(), os.path.join(out_path, "model"))
cfg = vars(args)
cfg["nr_features"] = train_dataset.num_feats
cfg["train_losses"] = train_losses
cfg["test_losses"] = test_losses
cfg["nr_inp"] = nr_inp
cfg["coord_dim"] = coord_dim
cfg["training_time"] = time.time() - start_training_time
with open(os.path.join(out_path, "cfg_res.json"), "w") as outfile:
    json.dump(cfg, outfile)
