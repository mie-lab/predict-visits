import numpy as np
import torch

from predict_visits.dataset import MobilityGraphDataset
from predict_visits.geo_embedding.embed import apply_log_coords

from torch_geometric.utils import from_scipy_sparse_matrix
import torch_geometric
import scipy.sparse as sp


def inverse_coord_normalization(log_coords):
    log_coords = log_coords * 10
    temp_sign = np.sign(log_coords)
    coords = temp_sign * (np.exp(np.absolute(log_coords)) - 1)
    return coords


def coord_normalization(coords):
    log_coords = apply_log_coords(coords) / 10
    return log_coords


class HomeLocationDataset(MobilityGraphDataset):
    def __init__(self, *args, nr_ref, **kwargs):
        self.nr_ref = nr_ref
        super().__init__(*args, **kwargs)
        # count the features without the coordinates
        self.num_feats = self.num_feats - 2

    def random_reference_points(self, coordinates):
        std_x, std_y = (
            np.std(coordinates[:, 0]),
            np.std(coordinates[:, 1]),
        )
        factor_std = 1
        min_x, max_x = (-1 * factor_std * std_x, factor_std * std_x)
        min_y, max_y = (-1 * factor_std * std_y, factor_std * std_y)
        ref_points = np.vstack(
            (
                np.random.randint(min_x, max_x, self.nr_ref),
                np.random.randint(min_y, max_y, self.nr_ref),
            )
        )
        return np.swapaxes(ref_points, 1, 0)

    def transform_coords(self, coords, ref_points):
        return np.hstack([coords - ref_p for ref_p in ref_points])

    def get(self, idx):
        adj = self.adjacency[idx]
        node_feat = self.node_feats[idx]
        label_col = node_feat[:, -1]
        home_normed_coords = node_feat[:, :2]

        # divide home node from rest
        home_node = np.where(np.sum(home_normed_coords, 1) == 0)[0][0]
        known_nodes = np.delete(np.arange(len(label_col)), home_node)

        # normalize features by a few new points
        assert np.all(home_normed_coords[home_node] == 0)
        ref_points = self.random_reference_points(home_normed_coords)
        transformed_coords = self.transform_coords(
            home_normed_coords, ref_points
        )

        # apply log and normalize
        transformed_coords = coord_normalization(transformed_coords)

        # concat with other node features (replacing the original coordinates)
        transformed_feats = np.hstack((transformed_coords, node_feat[:, 2:]))

        # restrict feats to known nodes and convert to torch
        known_node_feats = torch.from_numpy(
            transformed_feats[known_nodes]
        ).float()
        # new node features and label
        home_coords = torch.from_numpy(
            transformed_feats[home_node, : self.nr_ref * 2]
        ).float()

        # test whether backwards transform is correct
        np_test = inverse_coord_normalization(
            transformed_feats[home_node, : self.nr_ref * 2]
        )
        test_home_correct = np.array(
            [
                np_test[k * 2 : (k + 1) * 2] + ref_points[k]
                for k in range(self.nr_ref)
            ]
        )
        assert np.all(np.isclose(test_home_correct, 0))
        return (
            known_node_feats,
            home_coords,
            torch.from_numpy(ref_points).float(),
        )
        ### for graph processing:
        #         # reduce adjacency matrix to known nodes and preprocess
        #         adj = adj[known_nodes]
        #         adj = adj[:, known_nodes]

        #         if self.adj_is_symmetric:
        #             adj = adj + adj.T
        #         if self.adj_is_unweighted:
        #             adj = (adj > 0).astype(int)
        #         # transform to index & attr
        #         edge_index, edge_attr = from_scipy_sparse_matrix(adj)
        #         data_sample = torch_geometric.data.Data(
        #             x=known_node_feats,
        #             edge_index=edge_index,
        #             edge_attr=edge_attr.float(),
        #             y=home_coords,
        #         )

        #         adj_new = to_scipy_sparse_matrix(
        #           data_sample.edge_index, num_nodes=data_sample.num_nodes
        #         )
        #         num_components, component = sp.csgraph.connected_components(
        #               adj_new)
        #         _, count = np.unique(component, return_counts=True)
        #         subset = np.in1d(component, count.argsort()[-1:])
        #         print("NUM NODES largest component", sum(subset))
        #         return data_sample
        ###

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)
