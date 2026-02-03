"""
NAME:        src/models/gru.py
VERSION:     1.0
DESCRIPTION: GRU (Gated Recurrent Unit) Model for Battery Capacity Prediction
             
             GRU vs LSTM:
             - GRU has fewer parameters (no separate cell state)
             - Often trains faster (30-40% fewer computations)
             - Can perform similarly to LSTM on many tasks
             - Some research shows GRU better for battery degradation
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout):
        """
        Standard GRU for Time-Series Regression
        
        Differences from LSTM:
        - Single hidden state (no cell state)
        - Update gate + Reset gate (vs Input/Forget/Output gates)
        - ~30% fewer parameters than LSTM
        """

        super(GRUModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
        # GRU Layer
        # input_size=1, because 1 value per step across 120 steps
        # batch_first=True -> Input shape: (batch, Seq_len, Features)
        self.gru = nn.GRU(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
        )

        # The Head (Fully connected layer)
        # Maps final hidden state to a single output
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        """
        Input 'x' shape: (batch_size, 120) -> Q-V curve
        """
        
        # 1. Reshape for GRU: Need (Batch, Seq, Features)
        #       [Batch, 120] -> [Batch, 120, 1]
        x = x.unsqueeze(-1)

        # 2. Run GRU
        out, _ = self.gru(x)

        # 3. Take the LAST time step
        #       It summarises the whole curve
        last_step = out[:, -1, :]

        # 4. Predict
        prediction = self.fc(last_step)

        return prediction.squeeze() # Returns shape (Batch_Size)


