"""
This file contains:
    - Experimental Split of battery Cells for Training/Validation/Testing
    - An object to store of the battery cell data
"""

import torch
from torch.utils.data import Dataset
import os

# ==========================================================================
# --------              EXPERIMENTAL SPLITS (Stratified - All Protocols)  --------
# ==========================================================================

# STRATIFIED SPLIT: All protocols represented in train/val/test
# This prevents distribution shift and ensures fair evaluation

# Training Data (70%): Mixed protocols for robust learning
TRAIN_CELLS = [
    '01', '05', '06', '07', '09', '11', '12', '14', '15', '17', 
    '22', '23', '24', '25', '28', '30', '31', '33', '34', '35', 
    '36', '37', '39', '41', '42', '46', '47', '48', '49', '50', 
    '52', '54', '58', '59', '62', '63', '65', '66', '67', '68', 
    '69', '70', '71', '72', '74', '75', '77'
]

# Validation Data (15%): Mixed protocols for fair tuning
VAL_CELLS = [
    '18', '26', '27', '38', '40', '43', '53', '55', '57', '60', '64', '73'
]

# Test Data (15%): Mixed protocols for unbiased evaluation
TEST_CELLS = [
    '03', '04', '08', '20', '21', '29', '32', '44', '45', '51', '56', '61', '76'
]

# OLD SPLIT (Distribution mismatch - DO NOT USE):
# TRAIN_CELLS = ["01", "03", "05", ..., "40"]  # Only Rd_3C
# VAL_CELLS = ["04", "08", "42", "43"]  # Only Rd_3C
# TEST_CELLS = ["22", "32"]  # Only fixed protocols (causes 7× error!)


# ==========================================================================
# --------                  DATASET CLASS                      --------
# ==========================================================================

# MISSING LINE ADDED BELOW:
class BatteryDataset(Dataset):
    def __init__(self, root_dir, cell_ids, normalise=True):
        """
        Args:
            root_dir (str): Path to the 'tensors_qv' folder.
            cell_ids (list): List of cell ID strings (e.g. ['01', '03']).
            normalise (bool): If True, divides Capacity by 2.4Ah (Nominal).
        """
        self.inputs = []
        self.targets = []
        
        # 1. Iterate over requested Cell IDs
        found_files = 0
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Data directory not found: {root_dir}")

        all_files = os.listdir(root_dir)
        
        for cell_id in cell_ids:
            # Find file matching "01_*.pt" (e.g., "01_Rd_3C.pt")
            fname = next((f for f in all_files if f.startswith(f"{cell_id}_")), None)
            
            if fname is None:
                # Checking if it's strictly required or just a warning
                # print(f"Warning: Cell {cell_id} not found in {root_dir}")
                continue
                
            path = os.path.join(root_dir, fname)
            
            # 2. Load the .pt file
            try:
                # weights_only=False helps avoid future warning errors in some vers
                data = torch.load(path, weights_only=True)
                x_cell = data['X'] # Shape: (Num_Cycles, 120)
                y_cell = data['y'] # Shape: (Num_Cycles,)
                
                self.inputs.append(x_cell)
                self.targets.append(y_cell)
                found_files += 1
            except Exception as e:
                print(f"Error loading {fname}: {e}")

        # FIX 1: Indentation fixed (raise is now indented under if)
        # FIX 2: Moved OUTSIDE the loop. You want to check this after checking ALL files.
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
        # Returns: (Q-V Curve, Capacity)
        return self.X[idx], self.y[idx]
