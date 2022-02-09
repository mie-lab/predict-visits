import os
import torch
import argparse
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from predict_visits.dataset import MobilityGraphDataset
from predict_visits.model import ClassificationModel

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
    default=False,
    type=bool,
    help="Use logarithm on labels",
)
parser.add_argument(
    "-k",
    "--nr_keep",
    default=50,
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
    default=200,
    type=int,
    help="number of epochs",
)
args = parser.parse_args()

model_name = args.model
# data files must exist in directory data
train_data_files = ["t120_gc1_poi.pkl", "t120_yumuv_graph_rep_poi.pkl"]
# ["t120_yumuv_graph_rep_poi.pkl", "t120_gc2_poi.pkl", "t120_tist_toph100_poi.pkl", "t120_geolife_poi.pkl"]
test_data_files = ["t120_gc2_poi.pkl"]
learning_rate = args.learning_rate
nr_epochs = args.nr_epochs
batch_size = 1
# TODO: implement version with higher batch size (with padding)
evaluate_every = 1

# set up outpath
out_path = os.path.join("trained_models", model_name)
os.makedirs("trained_models", exist_ok=True)
os.makedirs(out_path, exist_ok=True)

# Train on GC1, GC2 and YUMUV
train_data = MobilityGraphDataset(train_data_files, **vars(args))
train_loader = DataLoader(train_data, shuffle=True, batch_size=1)

# Test on Geolife
test_data = MobilityGraphDataset(test_data_files, **vars(args))
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# Create model - input dimension is the number of features of nodes in the graph
# and the number of features of the new location
model = ClassificationModel(
    graph_feat_dim=train_data.num_feats,
    loc_feat_dim=train_data.num_feats - 1,
)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


train_losses, test_losses = [], []
for epoch in range(nr_epochs):
    epoch_loss = 0
    # TRAIN
    for i, (adj, node_feat, loc_feat, lab) in enumerate(train_loader):
        optimizer.zero_grad()

        # batch size is 1 --> currently we drop the batch axis
        out = model(node_feat[0].float(), adj[0], loc_feat[0].float())

        # MSE
        loss = torch.sum((out - lab) ** 2)
        loss.backward()
        # print(torch.sum(model.dec_hidden_2.weight.grad))
        # print(torch.sum(model.dec_out.weight.grad))
        optimizer.step()

        epoch_loss += loss.item()
    # print(" out example", round(out.item(), 3), round(lab.item(), 3))
    epoch_loss = epoch_loss / i * 100  # compute average

    # EVALUATE
    if epoch % evaluate_every == 0:
        with torch.no_grad():
            test_loss = 0
            for i, (adj, node_feat, loc_feat, lab) in enumerate(test_loader):
                out = model(node_feat[0].float(), adj[0], loc_feat[0].float())
                test_loss += torch.sum((out - lab) ** 2).item()
        test_loss = test_loss / i * 100  # compute average

        print(epoch, round(epoch_loss, 3), round(test_loss, 3))
        train_losses.append(epoch_loss)
        test_losses.append(test_loss)

# save model
torch.save(model.state_dict(), os.path.join(out_path, "model"))
# save config & results
cfg = vars(args)
cfg["train_losses"] = train_losses
cfg["test_losses"] = test_losses
with open(os.path.join(out_path, "cfg_res.json"), "w") as outfile:
    json.dump(cfg, outfile)

# plot losses
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train_losses, label="Train", c="blue")
ax.set_ylabel("Train losses", c="blue")
ax1 = ax.twinx()
ax1.plot(test_losses, label="Test", c="red")
ax1.set_ylabel("Test losses", c="red")
plt.savefig(os.path.join(out_path, "loss_plot.png"))
