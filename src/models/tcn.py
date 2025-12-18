"""
NAME:        src/models/tcn.py
VERSION:     1.0 
DESCRIPTION: TCN (Temporal Convolution Network) / 1D-CNN
             ML model, a parallelism comparison to LSTM
"""

import torch
import torch.nn as nn

class TCNModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 kernel_size,
                 dropout):
        super(TCNModel, self).__init__()

        # Block 1: Capture Low-level features (Sharp drops, noise)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, 
                      kernel_size=kernel_size, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2) # 120 -> 60
        )
        
        # Block 2: Capture Mid-level features (Knees, plateaus)
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2) # 60 -> 30
        )
        
        # Block 3: Capture High-level features (Overall Capacity trend)
        self.block3 = nn.Sequential(
            nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling (Squashes to [Batch, Hidden, 1])
        )
        
        # Regression Head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # Input x: [Batch, 120] -> Needs [Batch, Channel, Length]
        x = x.unsqueeze(1) 
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)

        return x.squeeze()
