"""
Enhanced dataset that predicts NEXT cycle capacity (not current)
This is the correct formulation for battery health prediction
"""

import torch
from torch.utils.data import Dataset
import os

# Keep the same cell splits
from .dataset import TRAIN_CELLS, VAL_CELLS, TEST_CELLS


class BatteryDatasetPredictive(Dataset):
    def __init__(self, root_dir, cell_ids, normalise=True, predict_next=True):
        """
        Args:
            root_dir (str): Path to the 'tensors_qv' folder.
            cell_ids (list): List of cell ID strings (e.g. ['01', '03']).
            normalise (bool): If True, divides Capacity by 2.4Ah (Nominal).
            predict_next (bool): If True, predict next cycle capacity (shifted by 1)
        """
        self.inputs = []
        self.targets = []
        self.predict_next = predict_next
        
        # 1. Iterate over requested Cell IDs
        found_files = 0
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Data directory not found: {root_dir}")

        all_files = os.listdir(root_dir)
        
        for cell_id in cell_ids:
            fname = next((f for f in all_files if f.startswith(f"{cell_id}_")), None)
            
            if fname is None:
                continue
                
            path = os.path.join(root_dir, fname)
            
            # 2. Load the .pt file
            try:
                data = torch.load(path, weights_only=True)
                x_cell = data['X']  # Shape: (Num_Cycles, 120)
                y_cell = data['y']  # Shape: (Num_Cycles,)
                
                if predict_next:
                    # CRITICAL FIX: Predict next cycle capacity
                    # Input: Q-V curve from cycle i
                    # Target: Capacity from cycle i+1
                    if len(x_cell) > 1:  # Need at least 2 cycles
                        x_cell = x_cell[:-1]  # Cycles 0 to N-2
                        y_cell = y_cell[1:]   # Cycles 1 to N-1 (shifted)
                
                self.inputs.append(x_cell)
                self.targets.append(y_cell)
                found_files += 1
            except Exception as e:
                print(f"Error loading {fname}: {e}")

        if found_files == 0:
            raise RuntimeError(f"No valid files loaded for cells: {cell_ids}")

        # 3. Concatenate all cells into one big tensor
        self.X = torch.cat(self.inputs, dim=0)
        self.y = torch.cat(self.targets, dim=0)
        
        # 4. Normalization (Nominal Capacity = 2.4 Ah)
        if normalise:
            self.y = self.y / 2.4

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Returns: (Q-V Curve from cycle i, Capacity from cycle i+1)
        return self.X[idx], self.y[idx]
