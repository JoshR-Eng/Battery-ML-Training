"""
NAME:        src/models/cnn_lstm.py
VERSION:     1.0
DESCRIPTION: CNN-LSTM Hybrid Model for Battery Capacity Prediction
             
             Architecture:
             1. CNN layers extract local voltage patterns from Q-V curve
             2. LSTM layers capture temporal degradation progression
             3. Fully connected head outputs capacity prediction
"""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_channels, kernel_size, 
                 lstm_hidden_size, lstm_layers, output_size, dropout):
        """
        CNN-LSTM Hybrid for Q-V Curve Analysis
        
        Args:
            input_size (int): Input features per timestep (1 for Q-V curves)
            cnn_channels (list): List of CNN channel sizes, e.g., [16, 32, 64]
            kernel_size (int): Kernel size for CNN layers
            lstm_hidden_size (int): Hidden size for LSTM
            lstm_layers (int): Number of LSTM layers
            output_size (int): Output size (1 for capacity)
            dropout (float): Dropout rate
        """
        super(CNNLSTMModel, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        
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
        # STAGE 2: LSTM Sequence Processing
        # Purpose: Capture temporal relationships across voltage levels
        # =================================================================
        
        # Input to LSTM: [Batch, SeqLen=120, Features=cnn_channels[-1]]
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        
        # =================================================================
        # STAGE 3: Prediction Head
        # Purpose: Map LSTM output to capacity prediction
        # =================================================================
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, output_size)
        )
    
    
    def forward(self, x):
        """
        Forward pass through CNN-LSTM
        
        Input x: [Batch, 120] - Q-V curve with 120 voltage points
        Output: [Batch] - Predicted capacity
        """
        
        # 1. Reshape for CNN: [Batch, 120] -> [Batch, 1, 120]
        #    (Batch, Channels, Length) for Conv1d
        x = x.unsqueeze(1)
        
        # 2. CNN Feature Extraction
        #    Output: [Batch, cnn_channels[-1], 120]
        cnn_features = self.cnn(x)
        
        # 3. Reshape for LSTM: [Batch, cnn_channels[-1], 120] -> [Batch, 120, cnn_channels[-1]]
        #    (Batch, SeqLen, Features) for LSTM
        cnn_features = cnn_features.permute(0, 2, 1)
        
        # 4. LSTM Processing
        #    Output: [Batch, 120, lstm_hidden_size]
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        
        # 5. Use last timestep output (or could use hidden state)
        #    Shape: [Batch, lstm_hidden_size]
        last_output = lstm_out[:, -1, :]
        
        # 6. Prediction
        #    Output: [Batch, 1] -> [Batch]
        prediction = self.fc(last_output)
        
        return prediction.squeeze()


