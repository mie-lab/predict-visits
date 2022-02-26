import numpy as np
import torch.nn as nn
import torch


class KNN:
    """
    k-nearest neighbor baseline to infer number of visits
    """

    def __init__(self, k=1, weighted=False):
        self.k = k
        self.weighted = weighted
        # TODO: distinguish geographically close in space / normalize features!

    def __call__(self, data):
        """
        Get closes feature vector in node_features and use their label
        """
        node_features = data.x
        assert len(node_features.shape) == 2
        # assert that only one batch
        # assert len(torch.unique(data.batch)) == 1
        assert len(data.y.shape) == 2
        new_location_features = data.y[:, :-1]

        feats_wo_labels = node_features[:, :-1]
        distance_to_feats = torch.mean(
            (feats_wo_labels - new_location_features) ** 2, axis=1
        )
        knn_inds = torch.argsort(distance_to_feats)[: self.k]
        if self.weighted:
            knn_dist = distance_to_feats[knn_inds]
            normed_knn_dist = knn_dist / torch.sum(knn_dist)
            avg_label = torch.sum(normed_knn_dist * node_features[knn_inds, -1])
        else:
            avg_label = torch.mean(node_features[knn_inds, -1])
        return avg_label
