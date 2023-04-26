import torch

from predict_visits.model.embedding_model import EmbeddingModel


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        num_feats,
        main_model,
        feature_embedding=1,
        embedding_hidden=32,
        model_cfg={},
        **kwargs
    ):
        super(ModelWrapper, self).__init__()
        self.feature_embedding = feature_embedding
        # the main ML model
        self.main_model = main_model(num_feats, **model_cfg)
        if feature_embedding:
            self.embedding_module = EmbeddingModel(
                num_feats - 1, embedding_hidden
            )

    def forward(self, x):
        if self.feature_embedding:
            x = self.embedding_module(x)
        x = self.main_model(x)
        return x
