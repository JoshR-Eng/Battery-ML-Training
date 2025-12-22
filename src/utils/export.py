"""
NAME:        src/utils/export.py
VERSION:     1.0
DESCRIPTION: Exports PyTorch models to ONNX format
"""

import torch
import torch.onnx
import os
import yaml
import sys

# Custom Code
sys.path.append(os.getcwd()) # All access to all scripts in working dir
from src.models import get_model

def export_model(model, filepath, device):

    # 1. Load weights
    model.eval()

    # 2. Create Dummy Input (The "Tracer")
    # Shape: [Batch_Size, Length] -> [1, 120]
    dummy_input = torch.randn(1, 120, requires_grad=True).to(device)

    # 3. Export
    print(f"Exporting to ONNX file type to '{filepath}'...")
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['capacity'],
        dynamic_axes={'input': {0: 'batch_size'}, 
                      'capacity': {0: 'batch_size'}}
        )
    print("\tSuccessfully Exported!")
