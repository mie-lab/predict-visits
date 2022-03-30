import torch.nn as nn
import torch
import math
from torch import Tensor
from predict_visits.model.transforms import TransformFF


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_feats,
        num_layers,
        dropout,
        nhead,
        dim_feedforward,
        historic_input=10,
        out_dim=1,
        **kwargs
    ) -> None:
        super(TransformerModel, self).__init__()
        self.num_feats = num_feats

        # transform torch geometric data into tensor (sequ_len x bs x feats)
        self.pre_transform = TransformFF(
            flatten=False, historic_input=historic_input
        )
        self.model = Transformer(
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            d_input=num_feats,
        )
        self.pos_encoder = PositionalEncoding(num_feats, dropout)

        # feel free to delete or change this layer:
        # final
        self.final_residual = nn.Sequential(
            nn.Linear(num_feats, num_feats),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(num_feats, num_feats),
        )
        self.final_norm = nn.LayerNorm(num_feats)
        self.final_layer = nn.Linear(num_feats, out_dim)

    def forward(self, data):
        x = self.pre_transform(data)
        # print(x.size())
        # >> (8, 11, 24) with the default parameters
        # corresponds to (batch_size, number input locations, number features)

        # 1. transformer for old_loc

        x = self.pos_encoder(x * math.sqrt(self.num_feats))

        # src_mask is to ensure the model does not look into the future,
        # but we do not need this, as we have no sequence info (?)
        # src_mask = self._generate_square_subsequent_mask(x.size(0)).to(device)

        # src_padding_mask is to ensure the padded areas are not influencing
        # the attention score when normalizing. If padding is 0, then:
        # src_padding_mask = (x == 0).transpose(0, 1).to(device)
        out = self.model(x, src_mask=None, src_padding_mask=None)

        # Note: the last location is the test location! (x[:, -1, :] are
        # the features of the test location)

        # 2. feed forward
        x = self.final_residual(out[-1])  # use last sequence element
        x = self.final_norm(x)
        x = self.final_layer(x)

        # output must be a single number per sample (tensor of size (8,1))
        # and should be normalized between 0 and 1
        out = torch.sigmoid(x)
        # print(out.size())
        # >> (8,1)
        return out

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


class Transformer(nn.Module):
    def __init__(
        self, num_layers, dropout, nhead, dim_feedforward, d_input
    ) -> None:
        super(Transformer, self).__init__()
        # encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_input,
            nhead=nhead,
            activation="gelu",
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            # batch_first=True, # only pytorch 1.8.1 onwards
        )
        encoder_norm = torch.nn.LayerNorm(d_input)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )

    def forward(self, input, src_mask, src_padding_mask) -> Tensor:
        """Forward pass of the network."""
        return self.encoder(
            input, mask=src_mask, src_key_padding_mask=src_padding_mask
        )


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            -torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )
