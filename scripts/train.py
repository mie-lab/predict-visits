import os
import torch
import argparse
import json
import time
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader

from predict_visits.dataset import MobilityGraphDataset
from predict_visits.baselines.simple_median import SimpleMedian
from single_model_eval import evaluate

from predict_visits.config import model_dict

# model name is desired one
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="gcn",
    type=str,
    help="Name of the model type to use",
)
parser.add_argument(
    "-l",
    "--log_labels",
    action="store_true",
    help="Use logarithm on labels",
)
parser.add_argument(
    "-k",
    "--nr_keep",
    default=60,
    type=int,
    help="graph size",
)
parser.add_argument(
    "-r",
    "--learning_rate",
    default=1e-4,
    type=float,
    help="learning rate",
)
parser.add_argument(
    "-e",
    "--nr_epochs",
    default=2000,
    type=int,
    help="number of epochs",
)
parser.add_argument(
    "-f",
    "--relative_feats",
    type=int,
    default=1,
    help="represent node features relative to new node",
)
parser.add_argument(
    "-b",
    "--batch_size",
    default=8,
    type=int,
    help="batch size",
)
parser.add_argument(
    "-c",
    "--label_cutoff",
    default=0.95,
    type=float,
    help="cutoff for labels",
)
parser.add_argument(
    "-z",
    "--embedding",
    default="simple",
    type=str,
    help="coordinate embedding",
)
parser.add_argument(
    "-i",
    "--historic_input",
    default=7,
    type=int,
    help="nr nodes of historic mobility to use (only relevant for non-gcn)",
)
parser.add_argument(
    "-s",
    "--save_name",
    default="model",
    type=str,
    help="Name under which to save model",
)
args = parser.parse_args()

cfg_model = model_dict[args.model]
NeuralModel = cfg_model["model_class"]
inp_transform = cfg_model["inp_transform"](**vars(args))
cfg_model["model_cfg"]["historic_input"] = args.historic_input

model_name = args.save_name

# data files must exist in directory data
train_data_files = ["t120_gc1_poi.pkl", "t120_yumuv_graph_rep_poi.pkl"]
# ["t120_yumuv_graph_rep_poi.pkl", "t120_gc2_poi.pkl", "t120_tist_toph100_poi.pkl", "t120_geolife_poi.pkl"]
test_data_files = ["t120_gc2_poi.pkl"]
learning_rate = args.learning_rate
nr_epochs = args.nr_epochs
batch_size = args.batch_size
# TODO: implement version with higher batch size (with padding)
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

# Train on GC1, GC2 and YUMUV
train_data = MobilityGraphDataset(train_data_files, device=device, **vars(args))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# Test on Geolife
test_data = MobilityGraphDataset(test_data_files, device=device, **vars(args))

# Create model - input dimension is the number of features
model = NeuralModel(train_data.num_feats, **cfg_model["model_cfg"]).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

r3 = lambda x: round(x * 100, 2)  # round function for printing

start_training_time = time.time()
train_losses, test_losses, bl_losses = [], [], []
for epoch in range(nr_epochs):
    epoch_loss = 0
    # TRAIN
    for i, data_geometric in enumerate(train_loader):
        # label is part of data.y --> visits to the new location
        lab = data_geometric.y[:, -1:]
        data = inp_transform(data_geometric)

        optimizer.zero_grad()

        out = model(data.to(device))
        # MSE
        loss = torch.sum((out - lab) ** 2)
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
            res_models = evaluate(
                {"trained": model, "median": SimpleMedian()},
                {"trained": inp_transform},
                test_data,
            )
            test_loss = res_models["trained"]

        print(epoch, r3(epoch_loss), r3(test_loss), r3(res_models["median"]))
        train_losses.append(epoch_loss)
        test_losses.append(test_loss)
        bl_losses.append(res_models["median"])


print("Finished training", time.time() - start_training_time)


# save model
torch.save(model.state_dict(), os.path.join(out_path, "model"))
# save config & results
cfg = vars(args)
cfg["nr_features"] = train_data.num_feats
cfg["train_losses"] = train_losses
cfg["test_losses"] = test_losses
cfg["bl_losses"] = bl_losses
cfg["training_time"] = time.time() - start_training_time
with open(os.path.join(out_path, "cfg_res.json"), "w") as outfile:
    json.dump(cfg, outfile)
