import torch
from torch import nn as nn, Tensor
from torch.nn.init import xavier_uniform_


class Bumformer(nn.Module):
    def __init__(
            self, num_features=24, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
            dropout=0.0, activation="relu"):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.dense_encoder = nn.Linear(self.num_features, self.d_model)
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.nhead,
                                          num_encoder_layers=self.num_encoder_layers,
                                          num_decoder_layers=self.num_decoder_layers,
                                          dim_feedforward=self.dim_feedforward)
        self.tgt = torch.rand((1, self.num_features, self.d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, entities: torch.Tensor, mask=None):
        """
            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
            src: (S, E)(S,E) for unbatched input, (S, N, E)
            (S,N,E) if batch_first=False or (N, S, E) if batch_first=True
            tgt: (T, E)(T,E) for unbatched input, (T, N, E)
            (T,N,E) if batch_first=False or (N, T, E) if batch_first=True
        """
        #emb = self.dense_encoder(entities)
        print(entities.size(), flush=True)
        emb = self.transformer(entities, torch.Tensor.Size([entities.size()[0], 1, self.d_model]), src_key_padding_mask=mask)
        return emb

    def __repr__(self):
        return f"Bumformer(d_model={self.d_model},n_layers={self.num_encoder_layers + self.num_decoder_layers}," \
               f"nhead={self.nhead})"
