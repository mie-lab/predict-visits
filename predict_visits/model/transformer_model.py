import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, num_feats, out_dim=1, **kwargs) -> None:
        super(TransformerModel, self).__init__()
        self.num_feats = num_feats  # by default 24

        # feel free to delete or change this layer:
        self.last_layer = nn.Linear(20, 1)

    def forward(self, x):
        print(x.size())
        # >> (8, 11, 24) with the default parameters
        # corresponds to (batch_size, number input locations, number features)

        # Note: the last location is the test location! (x[:, -1, :] are
        # the features of the test location)
        raise NotImplementedError

        # output must be a single number per sample (tensor of size (8,1))
        # and should be normalized between 0 and 1
        out = torch.sigmoid(last_layer)
        # print(out.size())
        # >> (8,1)
        return out
