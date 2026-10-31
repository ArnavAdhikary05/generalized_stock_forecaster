from __future__ import annotations

import torch
from torch import nn


class CandleTransformer(nn.Module):
    """Shared Transformer encoder; intentionally has no ticker-ID embedding."""

    def __init__(
        self,
        input_features: int,
        sequence_length: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.input_projection = nn.Linear(input_features, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        # TransformerEncoder clones one layer; reinitialize cloned linear weights independently.
        for layer in self.encoder.layers:
            for module in layer.modules():
                if isinstance(module, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(module.in_proj_weight)
                    if module.in_proj_bias is not None:
                        nn.init.zeros_(module.in_proj_bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError(f"Expected [batch, sequence, features], got {tuple(inputs.shape)}")
        x = self.input_projection(inputs)
        x = x + self.position_embedding[:, : x.shape[1], :]
        encoded = self.encoder(x)
        context = self.final_norm(encoded[:, -1, :])
        regression = self.regression_head(context)
        direction_logit = self.direction_head(context).squeeze(-1)
        return regression, direction_logit
