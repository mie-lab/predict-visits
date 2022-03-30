import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModel(torch.nn.Module):
    poi_original_size = 16
    poi_emb_size = 16

    def __init__(self, orig_size, hidden_size):
        """embeds in same size"""
        super(EmbeddingModel, self).__init__()
        self.emb_fc1 = nn.Linear(orig_size, hidden_size)
        self.emb_fc2 = nn.Linear(hidden_size, orig_size)
        self.norm = nn.LayerNorm(orig_size)
        self.emb_dropout = nn.Dropout(p=0.1)

    def embed(self, x):
        emb = self.emb_fc2(self.emb_dropout(F.gelu(self.emb_fc1(x))))
        emb = self.norm(x + emb)
        return emb

    def forward(self, data):
        # 1. Graph processing - node embeddings
        batch_size = data.y.size()[0]
        all_features = torch.cat((data.x[:, :-1], data.y[:, :-1]), dim=0)
        embedded_features = self.embed(all_features)

        embedded_features_x = embedded_features[:-batch_size]
        embedded_features_y = embedded_features[-batch_size:]

        data.x = torch.cat((embedded_features_x, data.x[:, -1:]), dim=1)
        data.y = torch.cat(
            (embedded_features_y, torch.zeros(batch_size, 1)), dim=1
        )
        return data
