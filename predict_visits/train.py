import os
import torch
from torch.utils.data import DataLoader

from dataset import MobilityGraphDataset
from model import ClassificationModel

learning_rate = 1e-10
nr_epochs = 100
batch_size = 1
# TODO: implement version with higher batch size (with padding)

train_data = MobilityGraphDataset(os.path.join("data", "train_data.pkl"))
train_loader = DataLoader(train_data, shuffle=True, batch_size=1)

test_data = MobilityGraphDataset(os.path.join("data", "test_data.pkl"))
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

model = ClassificationModel(input_feat_dim=4, second_input_dim=3)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


train_losses, test_losses = [], []
for epoch in range(nr_epochs):
    epoch_loss = 0
    # TRAIN
    for i, (adj, nf, pnf, lab) in enumerate(train_loader):
        optimizer.zero_grad()

        out = model(nf[0].float(), adj[0], pnf[0].float())

        loss = torch.sum((out - lab) ** 2)
        loss.backward()
        # print(torch.sum(model.dec_hidden_2.weight.grad))
        # print(torch.sum(model.dec_out.weight.grad))
        optimizer.step()

        epoch_loss += loss.item()
    print(" out example", round(out.item(), 3), round(lab.item(), 3))

    # EVALUATE
    with torch.no_grad():
        test_loss = 0
        for i, (adj, nf, pnf, lab) in enumerate(test_loader):
            out = model(nf[0].float(), adj[0], pnf[0].float())
            test_loss += torch.sum((out - lab) ** 2).item()

    print(epoch, round(epoch_loss, 3), round(test_loss, 3))
    train_losses.append(epoch_loss)
    test_losses.append(test_loss)
