
import torch
from torch import nn

class DepthFeatureGRU(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=1, num_classes=3, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])
