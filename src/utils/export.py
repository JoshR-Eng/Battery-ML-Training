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

def export_model(model, filepath, device, batch_size: int = 1):

    model.eval()

    # The dummy input traces the computation graph.
    # batch_size=1  → single-cell deployment (one Q-V curve per inference)
    # batch_size=32 → 32-cell pack deployment (32 Q-V curves in one GPU call)
    # NOTE: no dynamic_axes — TensorRT requires a static batch dimension to
    # avoid the Squeeze node ambiguity that breaks ONNX parsing.
    dummy_input = torch.randn(batch_size, 120, requires_grad=True).to(device)

    print(f"Exporting to ONNX (batch_size={batch_size}) -> '{filepath}' ...")
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['capacity'],
        )
    print("\tSuccessfully Exported!")
