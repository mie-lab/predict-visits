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
        if len(node_features.size()) == 2:
            avg_label = torch.median(node_features[:, -1])
        elif len(node_features.size()) == 3:
            avg_label = torch.median(node_features[:, :, -1], dim=1).values
        else:
            raise ValueError()
        return avg_label
