import torch
from torch import nn


class HomeNet(nn.Module):
    def __init__(self, input_size, output_size, ff_layers=[64, 128, 64, 32]):
        super(HomeNet, self).__init__()
        self.ff_layers = nn.ModuleList([nn.Linear(input_size, ff_layers[0])])
        for i in range(len(ff_layers) - 1):
            self.ff_layers.append(nn.Linear(ff_layers[i], ff_layers[i + 1]))
        # last layer
        self.ff_layers.append(nn.Linear(ff_layers[-1], output_size))

    def forward(self, x):
        for layer in self.ff_layers[:-1]:
            x = torch.relu(layer(x))
        # last layer without activation
        output = self.ff_layers[-1](x)
        return output
