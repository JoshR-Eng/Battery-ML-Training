"""
This file contains:
    - Experimental Split of battery Cells for Training/Validation/Testing
    - An object to store of the battery cell data
"""


import torch
from torch.utils.data import Dataset
import os

# ==========================================================================
# --------              EXPERIMENTAL SPLITS (Based on Paper)        --------
# ==========================================================================

# Training Data:
#   - Random Charge
#   - 3C Discharge
TRAIN_CELLS = [
    "01", "03", "05", "07", "09", "11", "14", "15",
    "17", "18", "20", "21", "24", "25", "27", "28",
    "30", "31", "33", "34", "36", "37", "39", "40"
]

# Validation Data:
#   - Random Charge
#   - 3C Discharge
VAL_CELLS = ["04", "08", "42", "43"]

# Edge Test Data:
#   - Fixed Discharge Profiles
TEST_CELLS = ["13", "22", "32"]


# ==========================================================================
# --------                  DATASET CLASS                      --------
# ==========================================================================
def __init__(self, root_dir, cell_ids, normalize=True):
        """
        Args:
            root_dir (str): Path to the 'tensors_qv' folder.
            cell_ids (list): List of cell ID strings (e.g. ['01', '03']).
            normalize (bool): If True, divides Capacity by 2.4Ah (Nominal).
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
                # weights_only=False helps avoid future warning errors in some versions
                # but for now standard load is fine
                data = torch.load(path)
                x_cell = data['X'] # Shape: (Num_Cycles, 120)
                y_cell = data['y'] # Shape: (Num_Cycles,)

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
        if normalize:
            self.y = self.y / 2.4

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Returns: (Q-V Curve, Capacity)
        return self.X[idx], self.y[idx]
