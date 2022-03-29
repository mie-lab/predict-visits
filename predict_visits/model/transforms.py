from matplotlib.pyplot import hist
import torch
from torch_geometric.utils import to_dense_batch


class TransformFF:
    def __init__(self, historic_input=5, flatten=True, **kwargs):
        self.historic_input = historic_input
        self.flatten = flatten

    def __call__(self, geometric_data):
        historic_mobility, mask = to_dense_batch(
            geometric_data.x, geometric_data.batch
        )
        batch_size = historic_mobility.size()[0]
        # # next part not needed because already sorted
        # top_inds = torch.argsort(
        #     historic_mobility[:, :, -1], dim=1
        # )[:, -self.historic_input:]
        # historic_top_locs = torch.stack(
        #     [historic_mobility[i, top_inds[i]] for i in range(batch_size)]
        # )
        historic_top_locs = historic_mobility[:, -self.historic_input :]

        # historic_top_locs_flat = historic_top_locs.reshape((-1))
        new_loc = torch.unsqueeze(geometric_data.y.clone(), 1)
        # padding for label
        new_loc[:, :, -1] = 0

        # concat
        loc_sequence = torch.cat((historic_top_locs, new_loc), axis=1)
        if self.flatten:
            loc_sequence = loc_sequence.reshape(batch_size, -1)
        else:
            loc_sequence = torch.transpose(loc_sequence, 1, 0)
        return loc_sequence


class NoTransform:
    def __init__(self, **kwargs):
        pass

    def __call__(self, geometric_data):
        return geometric_data
