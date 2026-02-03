"""
NAME:        src/models/cnn_gru.py
VERSION:     1.0
DESCRIPTION: CNN-GRU Hybrid Model for Battery Capacity Prediction
             
             Architecture:
             1. CNN layers extract local voltage patterns from Q-V curve
             2. GRU layers capture temporal degradation progression
             3. Fully connected head outputs capacity prediction
             
             Advantages over CNN-LSTM:
             - Fewer parameters (~25% less)
             - Faster training (~30% speedup)
             - May perform similarly or better
"""

import torch
import torch.nn as nn


class CNNGRUModel(nn.Module):
    def __init__(self, input_size, cnn_channels, kernel_size, 
                 gru_hidden_size, gru_layers, output_size, dropout):
        """
        CNN-GRU Hybrid for Q-V Curve Analysis
        
        Args:
            input_size (int): Input features per timestep (1 for Q-V curves)
            cnn_channels (list): List of CNN channel sizes, e.g., [16, 32, 64]
            kernel_size (int): Kernel size for CNN layers
            gru_hidden_size (int): Hidden size for GRU
            gru_layers (int): Number of GRU layers
            output_size (int): Output size (1 for capacity)
            dropout (float): Dropout rate
        """
        super(CNNGRUModel, self).__init__()
        
        self.gru_hidden_size = gru_hidden_size
        self.gru_layers = gru_layers
        
        # =================================================================
        # STAGE 1: CNN Feature Extraction
        # Purpose: Extract local voltage patterns (voltage drops, plateaus)
        # =================================================================
        
        cnn_layers = []
        in_channels = input_size
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # Keep sequence length same
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        
        # =================================================================
        # STAGE 2: GRU Sequence Processing
        # Purpose: Capture temporal relationships across voltage levels
        # Advantage: Fewer parameters than LSTM, often faster
        # =================================================================
        
        # Input to GRU: [Batch, SeqLen=120, Features=cnn_channels[-1]]
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        
        
        # =================================================================
        # STAGE 3: Prediction Head
        # Purpose: Map GRU output to capacity prediction
        # =================================================================
        
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size // 2, output_size)
        )
    
    
    def forward(self, x):
        """
        Forward pass through CNN-GRU
        
        Input x: [Batch, 120] - Q-V curve with 120 voltage points
        Output: [Batch] - Predicted capacity
        """
        
        # 1. Reshape for CNN: [Batch, 120] -> [Batch, 1, 120]
        x = x.unsqueeze(1)
        
        # 2. CNN Feature Extraction
        cnn_features = self.cnn(x)
        
        # 3. Reshape for GRU: [Batch, cnn_channels[-1], 120] -> [Batch, 120, cnn_channels[-1]]
        cnn_features = cnn_features.permute(0, 2, 1)
        
        # 4. GRU Processing
        gru_out, _ = self.gru(cnn_features)
        
        # 5. Use last timestep output
        last_output = gru_out[:, -1, :]
        
        # 6. Prediction
        prediction = self.fc(last_output)
        
        return prediction.squeeze()


