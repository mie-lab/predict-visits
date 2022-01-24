import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from predict_visits.dataset import MobilityGraphDataset
from predict_visits.model import ClassificationModel

# model name is desired one
model_name = "jan22"
# data files must exist in directory data
train_data_files = ["t120_gc1_poi.pkl"]
# ["t120_yumuv_graph_rep_poi.pkl", "t120_gc2_poi.pkl", "t120_tist_toph100_poi.pkl", "t120_geolife_poi.pkl"]
test_data_files = ["t120_gc2_poi.pkl"]
learning_rate = 1e-3
nr_epochs = 50
batch_size = 1
# TODO: implement version with higher batch size (with padding)

# Train on GC1, GC2 and YUMUV
train_data = MobilityGraphDataset(train_data_files)
train_loader = DataLoader(train_data, shuffle=True, batch_size=1)

# Test on Geolife
test_data = MobilityGraphDataset(test_data_files)
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

    # EVALUATE
    with torch.no_grad():
        test_loss = 0
        for i, (adj, node_feat, loc_feat, lab) in enumerate(test_loader):
            out = model(node_feat[0].float(), adj[0], loc_feat[0].float())
            test_loss += torch.sum((out - lab) ** 2).item()

    print(epoch, round(epoch_loss, 3), round(test_loss, 3))
    train_losses.append(epoch_loss)
    test_losses.append(test_loss)

os.makedirs("trained_models", exist_ok=True)
torch.save(model.state_dict(), os.path.join("trained_models", model_name))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train_losses, label="Train", c="blue")
ax.set_ylabel("Train losses")
ax1 = ax.twinx()
ax1.plot(test_losses, label="Test", c="red")
ax1.set_ylabel("Test losses")
plt.savefig(os.path.join("trained_models", model_name + ".png"))
plt.show()
