import torch

from predict_visits.dataset import MobilityGraphDataset


def transform_geometric_to_simple(geometric_data, nr_keep_final):
    historic_mobility = geometric_data.x
    top_inds = torch.argsort(historic_mobility[:, -1])[-nr_keep_final:]
    historic_top_locs = historic_mobility[top_inds]
    # historic_top_locs_flat = historic_top_locs.reshape((-1))

    new_loc = geometric_data.y.clone()
    # padding for label
    new_loc[:, -1] = 0

    # concat
    loc_sequence = torch.cat((historic_top_locs, new_loc), axis=0)
    label = geometric_data.y[:, -1]
    return loc_sequence, label


class LocationDataset(MobilityGraphDataset):
    def __init__(self, *args, nr_keep_final=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.nr_keep_final = nr_keep_final

    def get_num_feats(self):
        inp, _ = self[0]
        return inp.shape[0] * inp.shape[1]

    def get(self, idx):
        geometric_data = super().get(idx)
        loc_sequence, label = transform_geometric_to_simple(
            geometric_data, self.nr_keep_final
        )
        #         print(loc_sequence.size(), new_loc.size(), label)
        return loc_sequence, label
