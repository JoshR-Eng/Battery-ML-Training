import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout):
        """
        Standard LSTM for Time-Series Regression
        """

        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
        # LSTM Layer
        # input_size=1, becuase 1 value per step across 120 steps
        # batch_first=True -> Input shape: (batch, Seq_len, Features)
        self.lstm = nn.LSTM(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = num_layers,
                batch_first=True,
                dropout=dropout
        )

        # The Head (Fully connected layer)
        # Maps final hidden state to a single output
        self.fc = nn.Linear(hidden_size, output_size)



    def forward(self, x):
        """
        Input 'x' shape: (batch_size, 120) -> Q-V curve
        """
        
        # 1. Reshape for LSTM: Need (Batch, Seq, Features)
        #       [Batch, 120] -> [Batch, 120, 1]
        x = x.unsqueeze(-1)

        # 2. Run LSTM
        out, _ = self.lstm(x)

        # 3. Take the LAST time step
        #       It summarises the whole curve
        last_step = out[:, -1, :]

        # 4. Predict
        prediction = self.fc(last_step)

        return prediction.squeeze() # Returns shape (Batch_Size)
