import numpy as np
import torch.nn as nn
import torch


class SimpleMedian:
    """
    k-nearest neighbor baseline to infer number of visits
    """

    def __init__(self):
        pass

    def __call__(
        self,
        node_features: torch.tensor,
        adjacency: torch.tensor,
        new_location_features: torch.tensor,
    ):
        """
        Get closes feature vector in node_features and use their label
        """
        assert len(node_features.size()) == 2
        assert len(adjacency.size()) == 2
        assert len(new_location_features.size()) == 1

        avg_label = torch.median(node_features[:, -1])
        return avg_label
