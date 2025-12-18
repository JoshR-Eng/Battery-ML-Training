"""
NAME:        src/model/__init__.py
VERSION:     1.0
DESCRIPTION: Select ML Model
"""

import torch.nn as nn
from .lstm import LSTMModel
from .tcn import TCNModel

# ...Import other models when made

def get_model(config):
    model_name = config['model']
    model_params = config['models'][model_name]

    if model_name == "LSTM":
        return LSTMModel(
            input_size = model_params['input_size'],
            hidden_size= model_params['hidden_size'],
            num_layers = model_params['num_layers'],
            output_size= model_params['output_size'],
            dropout    = model_params['dropout']
        )

    elif model_name == "TCN":
        return TCNModel(
            input_size = model_params['input_size'],
            hidden_size= model_params['hidden_size'],
            kernel_size= model_params['kernel_size'],
            output_size= model_params['output_size'],
            dropout    = model_params['dropout']
        )

    else:
        raise ValueError(f"Model '{model_name}' not found")
