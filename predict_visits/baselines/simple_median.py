import numpy as np
import torch.nn as nn
import torch


class SimpleMedian:
    """
    k-nearest neighbor baseline to infer number of visits
    """

    def __init__(self, mean=False):
        self.mean=mean

    def __call__(self, data):
        """
        Get closes feature vector in node_features and use their label
        """
        node_features = data.x
        assert len(node_features.shape) == 2
        # assert that only one batch
        # assert len(torch.unique(data.batch)) == 1

        if self.mean:
            avg_label = torch.mean(node_features[:, -1])
        else: 
            avg_label = torch.median(node_features[:, -1])
        return avg_label
