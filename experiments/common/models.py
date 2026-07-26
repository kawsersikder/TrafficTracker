"""Forecasting models: LSTM / GRU / TCN.

All models take (batch, history, features) and output:
    task=cls -> (batch, 5) class logits
    task=reg -> (batch, 1) CI prediction
"""
from __future__ import annotations

import torch
from torch import nn


class RNNForecaster(nn.Module):
    def __init__(self, in_feats: int, out_dim: int, cell: str = "gru",
                 hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(in_feats, hidden, num_layers=layers,
                           batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class _CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, k: int = 3, dropout: float = 0.2):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(c_out)
        self.norm2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.down(x)
        h = nn.functional.pad(x, (self.pad, 0))
        h = self.drop(torch.relu(self.norm1(self.conv1(h))))
        h = nn.functional.pad(h, (self.pad, 0))
        h = self.drop(torch.relu(self.norm2(self.conv2(h))))
        return torch.relu(h + res)


class TCNForecaster(nn.Module):
    def __init__(self, in_feats: int, out_dim: int, channels: int = 64,
                 dilations=(1, 2, 4, 8), dropout: float = 0.2):
        super().__init__()
        blocks, c = [], in_feats
        for d in dilations:
            blocks.append(_CausalBlock(c, channels, d, dropout=dropout))
            c = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x.transpose(1, 2))  # (B, C, T)
        return self.head(h[:, :, -1])


def build_model(name: str, in_feats: int, out_dim: int) -> nn.Module:
    name = name.lower()
    if name in ("gru", "lstm"):
        return RNNForecaster(in_feats, out_dim, cell=name)
    if name == "tcn":
        return TCNForecaster(in_feats, out_dim)
    raise ValueError(f"unknown model '{name}' (expected gru | lstm | tcn)")
