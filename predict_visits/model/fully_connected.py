import torch.nn as nn
import torch
from predict_visits.model.transforms import TransformFF


class FullyConnectedModel(nn.Module):
    def __init__(
        self,
        num_feats,
        out_dim=1,
        layers=[128, 64],
        historic_input=5,
        final_act="sigmoid",
        **kwargs
    ):
        super(FullyConnectedModel, self).__init__()
        self.pre_transform = TransformFF(
            flatten=True, historic_input=historic_input
        )
        input_size = num_feats * (historic_input + 1)
        self.ff_layers = nn.ModuleList([nn.Linear(input_size, layers[0])])
        for i in range(len(layers) - 1):
            self.ff_layers.append(nn.Linear(layers[i], layers[i + 1]))
        # last layer
        self.ff_layers.append(nn.Linear(layers[-1], out_dim))
        # set activation function for last layer
        final_act_dict = {
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "none": lambda x: x,
        }
        self.final_act = final_act_dict[final_act]

    def forward(self, data):
        x = self.pre_transform(data)
        for layer in self.ff_layers[:-1]:
            x = torch.relu(layer(x))
        # last layer without activation
        output = self.ff_layers[-1](x)
        output = self.final_act(output)
        return output
